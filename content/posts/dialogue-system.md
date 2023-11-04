---
title: "Knowledge-based Dialogue System on AWS"
date: 2021-12-07T00:00:00+01:00
draft: false
---


# Deploy an End-to-End Chinese Knowledge-based Dialogue System in a MLOps fashion on AWS

Knowledge graph database structures data as nodes and vertex. It provides flexibility and allows diversity and heterogeneity in real-world data. However, managing and querying such databases requires professionality in understanding graph query language and the graph database itself. Luckily, natural language processing (NLP) technics can help both in creating graph databases and understanding users' queries. 

In this post, we will build a system heavily relied on NLP that can extract information from unstructured texts and interpret users' natural language queries as graph query languages. Moreover, we will develop this system in an MLOps manner such that it can automatically update itself to cope with changes in data and schema.

Walkthrough of this post:

- **Part I Algorithm**: Introduce the model used for knowledge extraction and natural question understanding.
- **Part II Deployment**: An overview of how the trained solution is deployed on AWS.
- **Part III Pipelines**: Implement two automated model training pipelines for knowledge extraction and natural question understanding.
- **Part IV MLOps**: Handcraft key MLOps components including code- and data-triggered re-training, model monitor and scheduled solution update.

## Part I: Algorithm

In this part, I will briefly introduce the algorithms for knowledge extraction and natural question understanding.

 ### Knowledge Extraction from Unstructured Data

The goal of knowledge extraction is to populate a knowledge graph with nodes and relations. We formulate this task as triplet extraction -- we extract (subject, predicate, object) pairs from given texts, where the subject and the object are entities; the predicate represents one kind of relation. The information extraction is carried out in a supervised manner, which means we have known text - triplets pairs to guide the system to extract similar relationships on unseen texts.

In a given text, there may exist multiple subjects. Each subject can also correspond to more than one predicate. Furthermore, each (subject, predicate) pair can correspond to many objects. Therefore, in essence, the extraction model first identifies subjects in a text, then identifies predicates and objects corresponding to each recognized subject. The model we used for Chinese information extraction adapts from *CasRel* proposed by [Wei, et al.](https://aclanthology.org/2020.acl-main.136/). The model is illustrated in *Figure 1*. It has the following steps:

1. **Tokenization**: Each character is assigned a fixed token. The given text is embedded by concatenating each of its character tokens.
2. **Contextualization**: A Bi-LSTM and a self-attention layer make the sentence aware of the context and encode it for further classification.
3. **Strongest signal recognition**: The most significant hidden state from max pooling is duplicated to the same length with the sentence, then concatenated on top of the sentence's hidden states.
4. **Subject prediction**: The concatenated hidden states are passed to a 1-dimension convolution layer and to linear layers to predicate the pointers to the head and tail positions of subjects. It is worth noting that the activation layer applied on linear layers is a Sigmoid function instead of SoftMax which is often used for subject classification, this allows the existence of multiple subjects.
5. **Subject integration**: What is left is to predict predicates and objects. Sentence hidden states from subject prediction steps are reused. We integrate subject embeddings by duplicating the head and tail of a subject and concatenating it on top of the sentence's hidden states.
6. **Object prediction**: The hidden states are then passed to a convolution layer and two linear layers to predict predicates and corresponding heads and tails of objects. The output size of the linear layer for each character is equivalent to the number of known predicates. This means, for each known predicate, we estimate possible pointers to the head and tail of objects according to the subject and the predicate.

![Model diagram for information extraction](https://imgur.com/doCvYAB.png) 

*Figure 1: Network for relation extraction*

### Question Understanding and Query Generation

In a given vertical application, those questions requiring querying graph databases are usually within certain categories. Thus we adopt a classification approach for question understanding. We first classify a natural language question into a pre-defined type. Some slots are associated with each type, i.e. you need to know the film name if you are asking about its director. The model for this part is *JointBERT* proposed by [Chen, et al.](https://arxiv.org/abs/1902.10909). It has a rather simple workflow compared to the last one:

1. **Tokenization**: Tokenize the given question with a BERT tokenizer. At the beginning of a sentence, a [CLS] token would be added, each rest character would be tokenized as a fixed ID.
2. **Embed with BERT**: Feed this to the BERT model which is consisted of 6 self-attention layers. Each token will have an embedded vector representation. Each vector actually has a full understanding of the whole sentence. (Check this blog for how it works: [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/))
3. **Classify intention and extract slots**: An intent classifier classifies logits of token [CLS] into a certain pre-defined intent. A slot classifier classifies each of the rest tokens in a *BIO* fashion. *B* signifies the start (**b**oundary) of a slot, *I* represents that the token is still a part of (**i**n) the previous slot, *O* means this token does not belong to (**o**ut) any slot. Each B label comes also with the type of the label, i.e. book, a film, a name.

![question understanding model](https://imgur.com/UitBE87.png)

*Figure 2: Question understanding model*

With this model, we can know the category and corresponding slot values of a question. We can then simply generate a graph query with a hand-crafted query template specific to each question category.

## Part II: Deployment

### Overview of the Deployment Architecture

Deployment of a machine learning-driven application is easy with AWS inference services. SageMaker *Endpoint* can host a managed inference cluster over multiple availability zones. Each of the instances within the cluster comes with a RESTful API for inference. A load balancer monitors endpoints' health status and passes on queries from other layers.

We have two machine learning models in our project, but only the question understanding model needs to be always on for question interpretation. The following architecture shows how each component of the system is joined together to interpret questions and retrieve answers.

1. A knowledge graph is generated and stored in the AWS Neptune graph database and is periodically updated.
2. SageMaker Endpoint automatically builds a scalable cluster that can host trained question understanding models. It uses RESTful APIs designed for the deep models to communicate with other components of the system.
3. An API Gateway stands and the edge of the system to receive requests from users and send back results.
4. An AWS Lambda function connects all other components. It first converts users' requests accordingly with endpoint APIs. Then it parses natural questions with SageMaker endpoints and populates a graph query. Finally, it retrieves answers by running the query on the Neptune database and returns results to API Gateway.

![architecture overview](https://imgur.com/PvrDxIl.png)

*Figure 3: Deployment Architecture*

## Part III: Pipelines -- The First Step Towards MLOps

SageMaker Pipeline is a CI/CD service that can compose, manage and reuse machine learning workflows. Amazon describes it as this [[3](https://aws.amazon.com/sagemaker/pipelines/)]:

> SageMaker Pipelines helps you automate different steps of the ML workflow, including data loading, data transformation, training and tuning, and deployment.... You can share and re-use workflows to recreate or optimize models, helping you scale ML throughout your organization.

SageMaker Pipelines migrate machine learning model building from updating codes in Jupyter notebooks to a collection of well-managed, standard, and structured steps like data preparation, training and deploying, etc. Steps are decoupled from each other, making them easy to manage. You can achieve cost efficiency and speed up training by running different steps on different computing instances according to their needs. Data dependencies between steps are established via step *properties*. SageMaker Pipelines then construct a directed acyclic graph (DAG) with these dependencies as shown below. Data transfer between steps is usually through AWS Simple Storage Service (S3).

![first pipeline](https://imgur.com/zvEgYCv.png)

*Figure 4. The first pipeline: Extract relations from unstructured texts, inference on all texts, and build a Neptune database*

### The First Pipeline: Construct Graph Database

In this first pipeline, we decompose the knowledge extraction task into independent steps as shown in *Figure 4*. Let's look at what different steps are doing:

- **Processing** - We first preprocess raw data from the dataset and store results onto the AWS S3. 
- **Train** - Training step then fetches data from S3, initiates extraction model, and starts training. 
- **EvaluateModel** - After training, the evaluation step evaluates model metrics on test data and saves evaluation reports to S3. 
- **F1Condition & AlertDevTeam** - A condition step will then retrieve these metrics and compare them with predefined criteria. If the criteria are met, it will continue to register model, create model, etc.; if failed, it executes another set of actions -- in our case, it will alert the development team for further investigation. 
- **CreateKgGenModel** - Model creation step stores model artifacts and inference code. 
- **KgRegisterModel** - This step registers model artifacts from the training step in the model registry. The model registry is a place one can easily manage and deploy all trained models. 
- **KgTransform** - KgTransform step performs batch transformation on all data and creates raw entities and relationships for the knowledge graph. 
- **RetriveOrCreateNeptuneDB** - Before loading edges and vertex into the Neptune database, we also need to confirm whether the specified database exists. This step checks the availability of the specified Neptune cluster. If the cluster exists, this step retrieves its information like endpoint. If the specified cluster does not exit, This step will create one with the given name and configuration. 
- **NeptuneBulkload** - Finally, we bulk load extracted information to the Neptune database.

I will further elaborate on the implementation details of some important steps.

#### Data Preprocess

First, we need to download the raw dataset. We use Baidu's [knowledge extraction dataset](https://ai.baidu.com/broad/introduction?dataset=dureader). Following the notebook `sagemaker-pipelines-project.ipynb` in the root directory of the project, we download the raw dataset and upload it to S3:

```bash
wget http://dataset-bj.cdn.bcebos.com/qianyan/DuIE_2_0.zip
aws s3 cp DuIE_2_0.zip "s3://{default-bucket}/ie-baseline/raw/DuIE_2_0.zip"
```

Each of these steps uses a custom script. The preprocess code locates at pipelines/kg/preprocess.py, basically, it extracts sentence and (subject, predicate, object) pairs from the original dataset which contains extra information. It also splits processed data into train, dev, and test sets.

We wrap this custom script into a step of our pipeline, the code for this locates at function `get_step_processing` in `pipelines/kg/pipeline.py`:

```python
processing_step = ProcessingStep(
    name="Processing",
    code=os.path.join(BASE_DIR, "preprocess.py"),
    processor=processor,
    inputs=processing_inputs,
    outputs=processing_outputs,
    job_arguments=[
        "--input-data",
        processing_inputs[0].destination, # /opt/ml/processing/ie/data/raw
    ],
)
```

#### Model Training and Evaluation

Models are trained using the knowledge extraction algorithm introduced in part 1. You can further check its implementation details at  `pipelines/kg/train.py`.

This script constructs a step with `sagemaker.workflow.steps.ProcessingStep`.  Details for the following snippets are at `get_step_training` in `pipelines/kg/pipeline.py`. We first define a *estimator* where the training script will be running:

```python
estimator = PyTorch(
    entry_point='train.py', # relative path to unzipped sourcedir
    source_dir=BASE_DIR,
    role=role,
    instance_type=train_instance_type, # ml.c5.4xlarge, ml.g4dn.4xlarge, etc
    instance_count=train_instance_count,
    framework_version='1.8.1', # PyTorch version
    py_version='py3',
    output_path=f"s3://{bucket}/ie-baseline/outputs",
    code_location=f"s3://{bucket}/ie-baseline/source/train", # where custom code will be uploaded 
    hyperparameters={
        'epochs': epochs,
        'use-cuda': True,
        'batch-size': batch_size,
        'learning-rate': learning_rate
    },
    metric_definitions = metric_definitions,
    debugger_hook_config=debugger_hook_config,
    profiler_config=profiler_config,
    rules=rules
)
```

We can also set up metrics to track the model performance of the model during training. The specified metrics will be captured from standard output logs and then be visualized on the training process panel in SageMaker:

```python
metric_definitions = [
    {'Name': 'eval:f1', 'Regex': 'f1: ([0-9\\.]+)'},
    {'Name': 'eval:prec', 'Regex': 'precision: ([0-9\\.]+)'},
    {'Name': 'eval:recall', 'Regex': 'recall: ([0-9\\./]+)'}
]
```

Finally, we define a training step with the estimator and other configurations defined:

```python
training_step = TrainingStep(
    name="Train",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=dependencies['step_process'].properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="application/json",
        ),
    },
    cache_config=cache_config,
)
```

valuation is wrapped with a `ProcessingStep`. Logics for evaluation are not too different from the steps mentioned above. After evaluation, this step saves results to S3 using [*PropertyFile*](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html) for future retrieval.

#### Automatic Model Quality Check with Condition Step

We pre-define some metrics for model quality check, i.e. we set the minimum f1 score to 0.6 in our case. We retrieve model evaluation results from the last step with [JsonGet](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html). If metrics shown in the evaluation report are below our defined criterion, this model is automatically denied. Otherwise, it continues to deployment.

```python
min_f1_value = params['min_f1_value']
evaluation_report = properties['evaluation_report']
minimum_f1_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=dependencies['step_evaluate'].name,
        property_file=evaluation_report,
        json_path="f1",
    ),
    right=min_f1_value,  # accuracy
)
minimum_f1_condition_step = ConditionStep(
    name="F1Condition",
    conditions=[minimum_f1_condition],
    if_steps=[
        dependencies['step_register'], 
        dependencies['step_create_model'],
        dependencies['step_transform'], 
        dependencies['step_create_db'],
        dependencies['step_bulkload']
    ],  # success, continue with model registration
    else_steps=[
        dependencies['step_alert']
    ],  # fail, end the pipeline
)
```

#### Model Creation and Registration

After training, we wrap the results into a standard SageMaker model. It usually contains model artifacts and inference code. We create this model with [CreateModel step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-create-model) as below:

```python
model = PyTorchModel(
    name=transform_model_name,
    model_data=dependencies['step_train'].properties.ModelArtifacts.S3ModelArtifacts,
    framework_version='1.3.1',
    py_version='py3',
    entry_point='inference.py'
)
create_inputs = CreateModelInput(
    instance_type=inference_instance_type,
    accelerator_type="ml.eia1.medium",
)
step_create_model = CreateModelStep(
    name="CreateKgGenModel",
    model=model,
    inputs=create_inputs,
)
```

Imagine you have trained your model with different hyperparameters and even different datasets for hundreds of rounds, it's hard to not mess things up. We want a central management portal for these trained models. The *model registry* is exactly what we need. The official [document](https://aws.amazon.com/about-aws/whats-new/2021/08/amazon-sagemaker-model-registry-inference-pipelines/?nc1=h_ls) describes model registry as *a central repository for cataloging models for production, managing model versions, associating metadata with models, managing approval statuses of models, and automating their deployment with CI/CD* . We register the trained artifacts into the model registry with a step specifically built for this purpose called [RegisterModel Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-register-model).

#### Offline Inference with BatchTransform

There exist two types of transformation -- online and offline. The main difference between them is whether you already know all data that will be examined. In the case you do, you collect all of them and do inference on them all at once. If you don't, you usually set up a server and do inference on incoming data in real-time. We will touch on the second scenario later in the second pipeline.

Here, since we already have all unstructured text data collected. We create a batch transformation job, specify its inputs, and put it into a [TransformStep](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-transform):

```python
transformer = Transformer(
    model_name=dependencies['step_create_model'].properties.ModelName,
    instance_type=transform_instance_type,
    instance_count=1,
    output_path=f"s3://{bucket}/{transform_output_prefix}",
)
step_transform = TransformStep(
    name="KgTransform", transformer=transformer, inputs=TransformInput(data=batch_data)
)
```

Transform result will be triplets of (subject, predicate, object). We store these triplets in an S3 location. Later, we will load them into a Neptune database.

#### Create a Neptune Database in a Step Function

Most of the before-mentioned steps are rather standard -- they are prepared by SageMaker so that we only need to do small modifications to every step. However, we can also define a totally new step to achieve our desired function on our own. We will show you how to do it here.

In this step, we are going to create a Neptune graph database, or retrieve one if it already exists. In essence, what each step does is simply run an algorithm in an AWS-managed container. We can write a database creation step by doing the same: prepare a script `createdb.py` that accepts some parameters, then wrap it in a `ProcessingStep` to allow it to run on an AWS-managed container.

The following steps elaborate how to prepare a Neptune database for loading data from S3, which is done in a later step function:

1. We specify the name of the Neptune cluster in parameters. We check whether a Neptune cluster with this name exists, if not, we create one.

   ```python
   def get_or_create_db_cluster(db_cluster_identifier):
       neptune = boto3.client('neptune')
       try:
           response = neptune.describe_db_clusters(DBClusterIdentifier=db_cluster_identifier)
           db_cluster = response['DBClusters'][0]
       except ClientError as e:
           if e.response["Error"]["Code"] != 'DBClusterNotFoundFault':
               raise e
           print(f"Neptune Cluster {db_cluster_identifier} does not exist.")
           print(f"Trying to create a Neptune Cluster with identifier {db_cluster_identifier}")
           response = neptune.create_db_cluster(
               DBClusterIdentifier=db_cluster_identifier, 
               Engine='neptune'
           )
           db_cluster = response['DBCluster']
       return db_cluster
   ```

2. A Neptune cluster ought to be a collection of database instances, ideally at least two for failure recovery. So if there is not any instance inside this cluster, we create some.

   ```python
   def get_or_create_db_instance(db_cluster_identifier, db_instance_suffix, db_instance_class):
       neptune = boto3.client('neptune')
       db_instance_identifier = f"{db_cluster_identifier}-{db_instance_suffix}"
       try:
           response = neptune.describe_db_instances(DBInstanceIdentifier=db_instance_identifier)
           db_instance = response['DBInstances'][0]
       except ClientError as e:
           if e.response["Error"]["Code"] != 'DBInstanceNotFound':
               raise e
           print(f"Trying to create a Neptune DB instance with identifier {db_instance_identifier}")
           response = neptune.create_db_instance(
               DBInstanceIdentifier=db_instance_identifier,
               DBInstanceClass=db_instance_class,
               Engine='neptune',
               DBClusterIdentifier=db_cluster_identifier,
           )
           db_instance = response['DBInstance']
       return db_instance
   ```

3. To allow Neptune database bulk-load knowledge from S3 later, we need to attach a role to allow Neptune service access the S3 bucket "on our behalf".

   ```python
   def get_or_create_loadfroms3_role(load_from_s3_role_name):
       iam = boto3.client("iam")
       s3_read_only_policy_arn = 'arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess'
   
       assume_role_policy_doc = {
           "Version": "2012-10-17",
           "Statement": [
               {
                   "Sid": "",
                   "Effect": "Allow", 
                   "Principal": {
                       "Service": [
                         "rds.amazonaws.com"
                       ]
                     },
                   "Action": "sts:AssumeRole"
               }
           ],
       }
       try:
           iam_role_loadfroms3 = iam.create_role(
               RoleName=load_from_s3_role_name,
               AssumeRolePolicyDocument=json.dumps(assume_role_policy_doc),
               Description="Allow Amazon Neptune to Access Amazon S3 Resources",
           )
           # attach s3 read only policy
           response = iam.attach_role_policy(
               RoleName=load_from_s3_role_name,
               PolicyArn=s3_read_only_policy_arn
           )
           print('Role:\n', iam_role_loadfroms3)
           print('Attach Policy Response:\n', response)
       except ClientError as e:
           if e.response["Error"]["Code"] == "EntityAlreadyExists":
               print("Role already exists")
               iam_role_loadfroms3 = iam.get_role(
                   RoleName=load_from_s3_role_name
               )
               print(iam_role_loadfroms3)
           else:
               print("Unexpected error: %s" % e)
       return iam_role_loadfroms3
   ```

#### Bulk load Knowledge from S3

Knowledge in a graph database is stored as vertices (entities) and edges (relations). When possible relations and entities are not bound to a predefined set, which implies we cannot extract knowledge from data with classification, we need to manually clean the extraction result and manually establish correspondence between extracted information and the knowledge graph, i.e. multiple different mentions may correspond to the same node in a knowledge graph. 

In our case, we have a known number of relations, but we have unbounded entities with known entity types. We process entities in a simple way -- entities with the same name correspond to the same node in the knowledge graph. You can find instructions on bulk loading into Neptune [here](https://docs.aws.amazon.com/neptune/latest/userguide/bulk-load.html).

Basically, we first transform the extracted data into a list of nodes and a list of relations. Then we store them onto S3 as two separate files. Finally, we load them into Neptune with the bulk [loader](https://docs.aws.amazon.com/neptune/latest/userguide/bulk-load.html) developed by AWS.

### The Second Pipeline: Question Answering

The second pipeline shows how to construct a question understanding system. It also decomposes the construction process into similar steps as below.

![second-pipeline](https://imgur.com/qeB9nIe.png)

​	*Figure 5. The second pipeline: Train a question understanding model*

- **Processing** - We load question templates and labeled natural language questions.
- **Train** - Load data from S3, train a model that can classify intention, and extract related labels.
- **Evaluate Model** - Evaluate model on a separate test dataset, log results and save them to S3.
- **IntentAndSlotCondition** - Retrieve evaluation report, compare intent classification and slot value extraction result with predefined metrics.
- **CreateQAModel** - Create a SageMaker model which contains trained model artifacts and inference code.
- **QARegisterModel** - Register the model created from the last step to the model registry.
- **AlertDevTeam** - Alert the development team by phone or email if the training process failed.

#### Question Answering Model Deployment

All steps mentioned above works similarly with those from the first pipeline. The most significant difference lies in inference. This question answering model does online inference -- we don't know what questions our users will ask. We have to launch machines to deploy these trained models. When a user issues a question, our servers accept it, analyze this question in real-time, retrieve answers from the graph database, and finally return the answer to the user.

Remember we registered a question answering model into the model registry? Before putting this model into the production environment, there will be a human involved check -- it would be a catastrophe if the all-automated pipeline directly present a flawed model to users. After we check every evaluation report and the model itself, we update the model's approval status to True. Details can be found [here](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-approve.html).

```python
sm.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved",  # Other options are Rejected and PendingManualApproval
)
```

Then we create a SageMaker endpoint with the model:

```python
sm.create_endpoint(
    EndpointName=pipeline_endpoint_name, EndpointConfigName=endpoint_config_name
)
```

Because we have provided an inference code along with the SageMaker model, the SageMaker endpoint automatically builds up a RESTful API for real-time inference on the model. Moreover, the SageMaker endpoint supports auto-scaling, we don't have to worry much about rush hour influx.

## Part IV: MLOps

In this part, we will handcraft several MLOps components using AWS SageMaker, CodeCommit, CodeBuild, and a variety of other AWS services. [AWS MLOps template](https://aws.amazon.com/solutions/implementations/aws-mlops-framework/) is a good way to jumpstart an MLOps project, yet some services included are not available worldwide, i.e. CodePipeline is not yet available in the China region. Also, many implementation details in the MLOps template are hidden or prescribed. To demystify MLOps deployment on AWS and allow a higher degree of flexibility, we implement three major components manually to form a minimum viable MLOps solution. all AWS services are integrated into the project with their Python API boto3.

We will introduce 3 major components of MLOps:

- **Automated Model Update with Git**
- **Automated Model Update for New Training Data** 
- **Monitor Model Quality and Run Scheduled Updates**

### Automated Model Update with Git

Similar to DevOps, automation in building the project is the starting point of a CI/CD cycle. In essence, whenever a commit is submitted to the main branch of our machine learning project repository, it will automatically create a pipeline to build the solution, train a model and check its validity.

The automated model building is triggered by monitoring a CodeCommit repository. The following steps elaborate its implementation.

#### 1. Create a CodeCommit Repository

CodeCommit is a managed source control service that hosts Git repositories. It can be easily integrated with many other AWS services. It functions similar to other Git-based source control platforms like GitHub and GitLab. If you have not yet had your codes on CodeCommit, simply create a repository from the AWS console and push your existing project with `$ git push 'https://git-codecommit.[region].amazonaws.com/v1/repos/[repository-name]`.

#### 2. Create a CodeBuild Project

When a new commit comes, we will build our solution, run it, and generate an updated pipeline. But where does this happen? CodeBuild is a continuous integration service from AWS, it compiles source code, runs tests, and generates deployable packages. In this step, we create a *target* with CodeBuild so that every time a commit comes, this target is triggered to process the rest steps. 

We first create a script that creates pipeline objects from their respective definition, takes arguments, and runs them. AWS MLOps template provides a universal script for this purpose, which can be found in [github.com/aws-samples/amazon-sagemaker-secure-mlops](https://github.com/aws-samples/amazon-sagemaker-secure-mlops/blob/master/mlops-seed-code/model-build-train/pipelines/run_pipeline.py). In our project, it is placed at `pipelines/run_pipeline.py`

CodeBuild essentially runs a user-defined set of commands to build the solution in a cloud environment. We specify this set of commands in a yaml file `codebuild-buildspec.yml`. Commands are separated into install and build phases. In install phase, we install dependencies that our project needs; in build phase, we invoke the `run_pipeline.py`  we defined previously and pass in required parameters.

```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.8
    commands:
      - pip install --upgrade --force-reinstall . awscli
      - pip install -r pipelines/kg/requirements.txt
      - pip install -r pipelines/qa/requirements.txt
  
  build:
    commands:
      - export PYTHONUNBUFFERED=TRUE
      - export SAGEMAKER_PROJECT_NAME_ID="${SAGEMAKER_PROJECT_NAME}-${SAGEMAKER_PROJECT_ID}"
      - |
        run-pipeline --module-name pipelines.kg.pipeline \
          --role-arn $SAGEMAKER_PIPELINE_ROLE_ARN \
          --tags "[{\"Key\":\"sagemaker:project-name\", \"Value\":\"${SAGEMAKER_PROJECT_NAME}\"}, {\"Key\":\"sagemaker:project-id\", \"Value\":\"${SAGEMAKER_PROJECT_ID}\"}]" \
          --kwargs "{\"region\":\"${AWS_REGION}\",\"sagemaker_project_arn\":\"${SAGEMAKER_PROJECT_ARN}\",\"role\":\"${SAGEMAKER_PIPELINE_ROLE_ARN}\",\"default_bucket\":\"${DEFAULT_BUCKET}\",\"pipeline_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"model_package_group_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"base_job_prefix\":\"${SAGEMAKER_PROJECT_NAME_ID}\"}"
      - |
        run-pipeline --module-name pipelines.qa.pipeline \
          --role-arn $SAGEMAKER_PIPELINE_ROLE_ARN \
          --tags "[{\"Key\":\"sagemaker:project-name\", \"Value\":\"${SAGEMAKER_PROJECT_NAME}\"}, {\"Key\":\"sagemaker:project-id\", \"Value\":\"${SAGEMAKER_PROJECT_ID}\"}]" \
          --kwargs "{\"region\":\"${AWS_REGION}\",\"sagemaker_project_arn\":\"${SAGEMAKER_PROJECT_ARN}\",\"role\":\"${SAGEMAKER_PIPELINE_ROLE_ARN}\",\"default_bucket\":\"${DEFAULT_BUCKET}\",\"pipeline_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"model_package_group_name\":\"${SAGEMAKER_PROJECT_NAME_ID}\",\"base_job_prefix\":\"${SAGEMAKER_PROJECT_NAME_ID}\"}"
      - echo "Create/Update of the SageMaker Pipeline and execution completed."
```

#### 3.Create an Event that triggers CodeBuild Task with CodeCommit

Now we have the target to trigger, we have not yet set up a trigger for it. The way we monitor the CodeCommit repository is similar to the [Publish–subscribe pattern](https://en.wikipedia.org/wiki/Publish–subscribe_pattern). Whenever there is a write event in CodeCommit, CodeCommit will send out an event to an account-wise event bus. An event log for a CodeCommit commit looks like this:

```json
{
    "version": "0",
    "id": "5ea772f9-3a21-2c14-96fb-d127deb3848d",
    "detail-type": "CodeCommit Repository State Change",
    "source": "aws.codecommit",
    "account": "[account-id]",
    "time": "2021-09-26T07:32:10Z",
    "region": "[region]",
    "resources": [
        "arn:aws:codecommit:[region]:[account-id]:sagemaker-CKGQA-p-kiqtyrraeiec-modelbuild"
    ],
    "detail": {
        "callerUserArn": "arn:aws:iam::[account-id]:user/[username]",
        "commitId": "45f8853bb3ef7910ada974a7a53ca14126cf4c84",
        "event": "referenceUpdated",
        "oldCommitId": "d65a5e3fac876f529f5fdc5e0273b79ee3d0bfae",
        "referenceFullName": "refs/heads/main",
        "referenceName": "main",
        "referenceType": "branch",
        "repositoryId": "[repositoryId]",
        "repositoryName": "sagemaker-CKGQA-p-kiqtyrraeiec-modelbuild"
    }
}
```

We only need to capture this event from the event but and then trigger our target. *EventBridge* is a serverless event bus managed by AWS. We first create a pattern to match the commit log above/ You need to specify the CodeCommit repository ARN and the monitored branch name:

```python
import json

codecommit_pattern = {
    "detail-type": ["CodeCommit Repository State Change"],
    "resources": [code_build_repository_arn],
    "source": ["aws.codecommit"],
    "detail": {
        "referenceType": ["branch"],
        "referenceName": [code_build_repository_branch]
    }
}
```

Then we create a rule which matches events with the pattern we defined.

```python
import boto3

events = boto3.client('events')

events.put_rule(
    Name=build_trigger_rule_name,
    EventPattern=codecommit_pattern_json,
    State="ENABLED",
    Description=build_trigger_rule_description,
    EventBusName="default",
    Tags=[
        {
            'Key': 'event',
            'Value': 'model-build-commited'
        },
    ],
)
```

Finally, we set the target as the CodeBuild task we defined previously:

```python
events.put_targets(
    Rule=build_trigger_rule_name,
    EventBusName='default',
    Targets=[
        {
            "Id": code_build_project_target_id,
            "Arn": code_build_project_arn,
            "RoleArn": sm_products_use_role_arn
        }
    ]
)
```

### Automated Model Update for New Training Data

Apart from an accurate model, up-to-date data is often more crucial for business competency. In our case, we will update the relation-extraction model when the schema is changed or when there are new labeled texts comes in. And we will update the question understanding model from user-fed inference data to keep the model aligned with the real-data distribution.

In this section, only training data are variable, we still preprocess them and train the model in the same manner. Therefore, we adopt a simple approach: monitor the data source, and retrain the model with updated data when the data source is updated.

We monitor training data change on S3 in the same logic: we first capture data change events and then trigger some target event. However, different from other AWS account activities, Object-level **data events** are not logged by default (otherwise your account event bus may be overwhelmed by data event logs). We need to use another AWS service called *CloudTrail* to help us log desired data events.

#### 1. Attach policy to S3 bucket to receive the log files

By default, Amazon S3 buckets and objects are private. Only the resource owner (the AWS account that created the bucket) can access the bucket and the objects it contains. To allow cloud trail log data events, we have to grant access permissions to CloudTrail by writing an access policy. Check [permission for cloudtrail](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/create-s3-bucket-policy-for-cloudtrail.html?icmpid=docs_cloudtrail_console) for more details. 

First, we define a bucket policy to allow CloudTrail to write captured logs into it:

```python
import boto3
account_id = boto3.client('sts').get_caller_identity().get('Account')
log_bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AWSCloudTrailAclCheck20150319",
            "Effect": "Allow",
            "Principal": {"Service": "cloudtrail.amazonaws.com"},
            "Action": "s3:GetBucketAcl",
            "Resource": f"arn:aws:s3:::{default_bucket}"
        },
        {
            "Sid": "AWSCloudTrailWrite20150319",
            "Effect": "Allow",
            "Principal": {"Service": "cloudtrail.amazonaws.com"},
            "Action": "s3:PutObject",
            "Resource": f"arn:aws:s3:::{default_bucket}/AWSLogs/{account_id}/*",
            "Condition": {"StringEquals": {"s3:x-amz-acl": "bucket-owner-full-control"}}
        }
    ]
}
log_bucket_policy = json.dumps(log_bucket_policy)
```

Then we attach this policy to the bucket where we want to save data event logs:

```python
s3 = boto3.client('s3')
s3.put_bucket_policy(Bucket=default_bucket, Policy=log_bucket_policy)
```

#### 2. Create a trail to log S3 events

S3 will not send out an object-level data event log by itself. We log writing events to a specific location by logging S3 `PUT` API call. A *trail* captures API calls and related events in your account and then delivers the log files to an S3 bucket that you specify. 

```python
cloudtrail = boto3.client('cloudtrail')
cloudtrail.create_trail(
    Name=trail_name,
    S3BucketName=default_bucket, # this specifies the bucket to save logs
    TagsList=[
        {
            'Key': 'event',
            'Value': 'kg-dataset-update'
        }
    ]
)
```

#### 3. Define an event selector for CloudTrail

There are usually spontaneously numerous API calls to your S3 buckets, i.e. from other services. An *event selector* can specify management and data event settings for the trail. If an API call matches any event selector, the trail processes and logs the event.

We create an event selector as follows to match `WriteOnly` API calls to the specified S3 bucket and prefix, i.e. `DefaultBucket/KnowledgeGraphTrainData`

```python
watched_s3_resource_arn = "arn:aws:s3:::{}/{}".format(watched_bucket, watched_prefix)
event_selector = [
    { 
        "ReadWriteType": "WriteOnly", 
        "IncludeManagementEvents":False, 
        "DataResources": 
            [
                { 
                    "Type": "AWS::S3::Object", 
                    "Values": [watched_s3_resource_arn] 
                }
            ]
    }
]
```

Then we attach this event selector to the trail we created:

```python
cloudtrail.put_event_selectors(
    TrailName=trail_name,
    EventSelectors=event_selector
)
```

The trail will not start logging until we manually activate it:

```python
cloudtrail.start_logging(
    Name=trail_name
)
```

#### 4. Create an EventBridge rule that can trigger SageMaker pipeline.

Till now, we have a working trail that will log `PUT` events to the location where we store training data. The next step is to capture this log in the same way as monitoring CodeCommit repository updates. 

An S3 API `PUT` call logged by CloudTrail sent to EventBridge bus looks like this (it's a hassle to capture this log by hand, you can check this [tutorial](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-log-s3-data-events.html) if you are interested):

```json
{
    "version": "0",
    "detail-type": "AWS API Call via CloudTrail",
    "source": "aws.s3",
    "detail": {
        ...
        "eventSource": "s3.amazonaws.com",
        "eventName": "PutObject", # PutObject event
        "requestParameters": { # watched bucket and prefix
            "bucketName": "[watched-bucked]",
            "Host": "[watched-bucket].s3.amazonaws.com",
            "key": "[prefix]"
        },
        ...
        "readOnly": false,
        "resources": [
            {
                "type": "AWS::S3::Object",
                "ARN": "arn:aws:s3:::[watched-bucket]/[prefix]"
            },
            {
                "accountId": "[account-id]",
                "type": "AWS::S3::Bucket",
                "ARN": "arn:aws:s3:::[watched-bucket]"
            }
        ],
        "eventType": "AwsApiCall",
        "managementEvent": false,
        "recipientAccountId": "[account-id]",
        "eventCategory": "Data" # data event
    }
}
```

We create an event pattern to match this event format:

```python
pattern = {
    "source": ["aws.s3"],
    "detail-type": ["AWS API Call via CloudTrail"],
    "detail": {
        "eventSource": ["s3.amazonaws.com"],
        "eventName": ["PutObject", "CompleteMultipartUpload", "CopyObject"],
        "requestParameters": {
            "bucketName": ["{}".format(watched_bucket)],
            "key": [watched_prefix]
        },
    },
}
pattern_json = json.dumps(pattern)
```

Then we create a rule with this pattern to watch data update on desired bucket and prefix:

```python
import boto3

events = boto3.client('events')

events.put_rule(
    Name=s3_rule_name,
    EventPattern=pattern_json,
    State="ENABLED", # enable by default
    Description=s3_rule_description,
    EventBusName="default",
    Tags=[
        {
            'Key': 'event',
            'Value': 'kg-dataset-update'
        },
    ],
)
```

#### 5. Add the pipeline as a target to the rule

Since only data is changed, we can follow the same pipeline that has already been built to retrain machine learning models. In this step, we set a known pipeline with ARN `pipeline_arn` as the target that will be executed whenever training data is updated.

```python
events.put_targets(
    Rule=s3_rule_name,
    EventBusName='default',
    Targets=[
        {
            "Id": pipeline_id, # arbitrarily define a unique id is enough
            "Arn": pipeline_arn,
            "RoleArn": run_pipeline_role_arn
        }
    ]
)
```

#### 6. Trigger this rule by writing new data

To check whether a data event is logged, go to the CloudTrail console,  select the trail you created. At the CloudWatch Logs section, create a  CloudWatch group for this trail. Then, go to CloudWatch and check the  stream logs of this group.

Say, we have already reduced data size to half the original items locally and verified the data, then we upload it:

```bash
# upload modified data to the watched location
$ aws s3 cp modified_dataset.zip s3://$watched_bucket/original_dataset.zip
```

Now go to the CloudTrail console and EventBridge console to check whether there is something happening! And check whether there is a new pipeline running as well! If everything goes well, well, we revert the pseudo data to its original version.

```bash
$ aws s3 cp original_dataset.zip s3://$watched_bucket/original_dataset.zip
```

### Monitor Model Quality and Run Scheduled Updates

When the machine learning model is deployed in production, we need to continuously monitor its quality and get notified. Early and proactive detection of model deviations enables us to take in-time actions, such as re-training models or adjusting model structures. There are mainly two types of monitor tasks. One is to monitor data quality, which aims at detecting drifts in data; another is to monitor model quality, which aims at detecting drift in model quality metrics, such as accuracy. Check [Amazon SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html) for more explanation. In our case, we process text data, there is no universal standard that can measure drifts in text data drifts. Thus we focus on monitoring model quality.

We monitor the quality of a model by comparing the predictions that the model makes with the actual ground truth labels that the model attempts to predict. To do this, we will collect real-time inference data, label them, and compare the labels with prediction results. We follow these steps to monitor model quality and update our model accordingly.

- **Enable data capture**: This allows Endpoint to capture real-time inference data and store them on S3.
- **Create a baseline**: A baseline job automatically creates baseline statistical rules and constraints to evaluate model performance. It takes in a baseline dataset, its ground-truth labels, and the model's predictions. 
- **Define and schedule model quality monitoring jobs**: Periodically evaluate model performance with respect to the baseline.
- **Ingest captured inference data**: We label captured inference data and merge them into our training dataset.

#### 1. Enable catching inference data on endpoints

To achieve this, the first step is to log inference data. Remember that we have created a trained model that contains model artifacts and inference code, assume the name of it is stored in `created_model_name`. We will configure this model to log a certain ratio of incoming inference data, then deploy it on an Endpoint.

Here we create an endpoint configuration that SageMaker hosting services uses to deploy models. `ProductionVariants` specifies the deployment environment and the model name. The `DataCaptureConfig` specifies logging requirements, including the percentage of data to log, where to save logged data, and whether to log both incoming queries and output responses.

```python
sm = boto3.Session().client(service_name="sagemaker", region_name=region)

endpoint_config_name = "qa-model-from-registry-epc-{}".format(timestamp)

create_endpoint_config_response = sm.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": "ml.m5.4xlarge",
            "InitialVariantWeight": 1,
            "InitialInstanceCount": 1,
            "ModelName": created_model_name,
            "VariantName": "AllTraffic",
        }
    ],
    DataCaptureConfig={
        'EnableCapture': True,
        'InitialSamplingPercentage': 10, # log 10% of total incoming queries
        'DestinationS3Uri': f"s3://{data_capture_bucket}/{data_capture_prefix}",
        'CaptureOptions': [ # log both incoming queries and output responses
            {
                'CaptureMode': 'Input'
            },
            {
                'CaptureMode': 'Output'
            },
        ]
    }
)
```

Then we deploy the model to an endpoint with the config above:

```python
endpoint_name = 'arbitrary_endpoint_name'
sm.create_endpoint(
    EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
)
```

Now, when new queries come to our endpoint, 10% of them will be logged for future labeling and improving model accuracy.

#### 2. Create a model quality baseline

A baseline job compares model predictions with ground truth labels in a baseline dataset. It generates a baseline that contains metrics and constraints for the future model quality monitor. We use `ModelQualityMonitor` provided by SageMaker to set up a baseline job.

First, we create an instance of the `ModelQualityMonitor` class

```python
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

model_quality_monitor = ModelQualityMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.4xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sess
)
```

Then we create a baseline dataset in JSON format that contains data, prediction, and ground truth labels. Here we utilize the predicted labels from the steps before.

```python
import panda as pd

with open('processed/baseline/seq.in') as f:
    x_input = f.readlines()
    x_input = [x.strip() for x in x_input]
with open('processed/baseline/label') as f:
    y_output = f.readlines()
    y_output = [y.strip() for y in y_output]

test_dataset = {
    'seq_in': x_input,
    'predicted_label': predicted_cls,
    'label': y_output
}

pd.DataFrame(test_dataset).to_json(path_or_buf='baseline_dataset.json', orient='records', lines=True, force_ascii=False)
```

We upload this baseline dataset to S3:

```bash
$ aws s3 cp baseline_dataset.json s3://[baseline-stored-bucket]/baseline_dataset.json
```

Now call the suggest_baseline method of the `ModelQualityMonitor` object to  run a baseline job. We need a baseline dataset that contains both  predictions and labels stored in Amazon S3.  Suggest baseline specification: 

```python
baseline_job = model_quality_monitor.suggest_baseline(
    job_name=baseline_job_name,
    baseline_dataset='s3://[baseline-stored-bucket]/baseline_dataset.json', # The S3 location of the validation dataset.
    dataset_format=DatasetFormat.json(lines=True), # Whether the file should be read as a json object per line
    output_s3_uri=baseline_job_output_s3, # The S3 location to store the results.
    problem_type='MulticlassClassification',
    inference_attribute= "predicted_label", # The column in the dataset that contains predictions.
    ground_truth_attribute= "label" # The column in the dataset that contains ground truth labels.
)
```

After the baseline job finishes, you can see the constraints that the  job generated. The constraints are thresholds for metrics that model monitor measures. If a metric goes beyond the suggested threshold, Model Monitor reports a violation. To view the constraints that the baseline job generated, call the `suggested_constraints` method of the baseline job.

```python
import pandas as pd
pd.DataFrame(baseline_job.suggested_constraints().body_dict['multiclass_classification_constraints']).T
```

The constraints for our model (only intention label) looks like this:

|                    | threshold | comparison_operator |
| ------------------ | --------- | ------------------- |
| accuracy           | 0.938547  | LessThanThreshold   |
| weighted_recall    | 0.938547  | LessThanThreshold   |
| weighted_precision | 0.939565  | LessThanThreshold   |
| weighted_f0_5      | 0.939081  | LessThanThreshold   |
| weighted_f1        | 0.938622  | LessThanThreshold   |
| weighted_f2        | 0.938475  | LessThanThreshold   |

This constraints is useful when we create a scheduled monitor job.

#### 3. Define and schedule model quality monitoring jobs

In this step, we create a model monitoring schedule for the endpoint created  earlier, using the baseline resources (constraints and statistics) to  compare against the real-time traffic. 

SageMaker model monitors model quality by analyzing data collected from an endpoint during a given period. What it does is to compare the collected dataset with the baseline constraints. The collected inference data are unlabeled. As a result, the comparison can only be done after we label the collected data. To address this, use offsets. Model quality jobs include `StartOffset` and `EndOffset` fields, they delay the analysis of collected data for a given period of time by when we should have finished data labeling. Examples of usages can be found [here](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-schedule.html).

First, we specify the endpoint to monitor:

```python
endpoint_input = sagemaker.model_monitor.EndpointInput(
    endpoint_name=pipeline_endpoint_name,
    destination='/opt/ml/processing/test_dataset.json',     # The destination of the input.
    inference_attribute='predicted_label', # JSONpath to locate predicted label(s)
    start_time_offset='-P8D',              # Monitoring start time offset, delays for 8 days
    end_time_offset='-P1D'				# Delay for 1 day
)
```

We can create such a schedule by calling [CreateMonitoringSchedule](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateMonitoringSchedule.html) API. Here we create a schedule that monitors the `endpoint_input` above. It runs daily. It uses the constraints from the baseline we created above by calling `model_quality_monitor.suggested_constraints()`.  When it finishes, a report will be generated and uploaded to `s3_report_path`.

```python
from sagemaker.model_monitor import CronExpressionGenerator
from time import gmtime, strftime

mon_schedule_name = 'qa-model-monitor-schedule-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model_quality_monitor.create_monitoring_schedule(
    monitor_schedule_name=mon_schedule_name,
    endpoint_input=endpoint_input,
    output_s3_uri=s3_report_path,
    problem_type='MulticlassClassification',
    constraints=model_quality_monitor.suggested_constraints(),
    ground_truth_input=f"s3://{ground_truth_bucket}/{ground_truth_prefix}/",
    schedule_cron_expression=CronExpressionGenerator.daily(),
    enable_cloudwatch_metrics=True,
)
```

#### 4. Ingest captured inference data

Recall that model quality monitors compare predictions metrics like accuracy with baseline constraints. For this to work, we need to periodically label captured data and upload them to S3. Moreover, we can re-feed these labeled data to the system for re-training the model, to enable the model to always have an "up-to-date" performance.

To let the model monitor recognize labeled data, the labels need to be in the format of:

```json
{
  "groundTruthData": {
    "data": "1",
    "encoding": "CSV" # only CSV supported at launch, we assume "data" only consists of label
  },
  "eventMetadata": {
    "eventId": "aaaa-bbbb-cccc"
  },
  "eventVersion": "0"
}
```

And the labels need to be stored at `s3://bucket/prefix/yyyy/mm/dd/hh`, where the date time is the time that the ground truth data is collected.

To label collected data, we first need to download them from S3. Each `capture_file` below is a `jsonl` file that contains a small amount of data captured during a certain period:

```python
import boto3

s3 = boto3.Session().client('s3')
current_endpoint_capture_prefix = '{}{}'.format(data_capture_prefix, pipeline_endpoint_name)
result = s3.list_objects(Bucket=data_capture_bucket, Prefix=current_endpoint_capture_prefix)
capture_files = [capture_file.get("Key") for capture_file in result.get('Contents')]

```

We download them from S3 with `S3Downloader`

```python
from sagemaker.s3 import S3Downloader
traffic = S3Downloader.read_file(f"s3://{data_capture_bucket}/{capture_files[0]}")
```

`traffic` is a line separated file, each line contains a piece of captured data in the following format:

```json
{
   "captureData":{
      "endpointInput":{
         "observedContentType":"text/csv",
         "mode":"INPUT",
         "data":"[incoming-data-payload]",
         "encoding":"BASE64"
      },
      "endpointOutput":{
         "observedContentType":"application/json",
         "mode":"OUTPUT",
         "data":"[predicted-result-payload]",
         "encoding":"BASE64"
      }
   },
   "eventMetadata":{
      "eventId":"4651a1bf-4b00-4248-a309-4a18b11e4277",
      "inferenceTime":"2021-09-27T03:47:25Z"
   },
   "eventVersion":"0"
}
```

The `eventId` is the key component to match our manually created labels with collected data and predictions.

Assume the questions we collect are:

```
['三度诱惑的导演是谁',
 '何藩导演了哪些电视剧',
 '我乔布斯是谁写的',
 '妖精凄卟曲之美男就地扑倒的作者是谁',
 '许晋亨的配偶是谁',
 '李嘉欣和谁结婚了',
 '决胜是谁的作品',
 '林晓蔚毕业于哪里']
```

We manually assign label for each of them:

```python
labels = ['ask_director', 'ask_films', 'ask_author', 'ask_author', 'ask_wife', 'ask_husband', 'ask_author', 'ask_school']
```

We can create a ground truth label in the following format. As shown below, `event_id` is the key to associating this to the collected data.

```python
ground_truth_data = {
    "groundTruthData": {
        "data": ','.join(labels),
        "encoding": "CSV" # only CSV supported at launch, we assume "data" only consists of label
    },
    "eventMetadata": {
        "eventId": event_id
    },
    "eventVersion": "0"
}
```

The upload URI can be generated as:

```python
from datetime import datetime

now = datetime.today()
ground_truth_upload_uri = f"s3://{ground_truth_bucket}/{ground_truth_prefix}/{now.year}/{now.strftime('%m')}/{now.strftime('%d')}/{now.strftime('%H')}/"
```

We then write this ground truth to a local file and upload it to S3:

```python
with open('output/ground_truth.json', 'w') as f:
    json.dump(ground_truth_data, f, ensure_ascii=False)
```

```bash
!aws s3 cp output/ground_truth.json $ground_truth_upload_uri
```

Till now, this data is ready for the model monitor to utilize, and even future re-training.

## Conclusion

In this post, we built an end-to-end knowledge-based question answering system. We first introduced its algorithms, then we illustrated how it is deployed on AWS. We spent a great amount of time showing how pipelines are built, and how a minimum MLOps project is handcrafted based on the pipelines. MLOps will continue to evolve and gain wider adoption. [AWS MLOps Framework](https://aws.amazon.com/solutions/implementations/aws-mlops-framework/) can give you further and detailed information on other aspects of MLOps offerings from AWS.
