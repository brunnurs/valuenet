# Hack Zurich Tutorial

Hi there! You can use ValueNet for Text-to-SQL translations in the following ways:
* You just wanna use the deployed **Text-to-SQL API** on the goverment data? Follow the instructions under [Use the deployed API](#use-the-deployed-api) (*easy*)
* You wanna **run the trained Text-to-SQL model locally**, connected to your own data? Follow the instruction under [Local Inference](#local-inference) (*medium, small GPU required*)
* You wanna **re-train the Text-to-SQL model with your own data**, maybe even **improve it**? Follow the instructions under [Train Model](#train-model) (*hard, powerful GPU required*)

## How does ValueNet work under the hood?
ValueNet is a neural network architecture which aims to translate a natural language question into the corresponding SQL query. 

Have a look at our paper for the details
[https://arxiv.org/abs/2006.00888](https://arxiv.org/abs/2006.00888)


## Use the deployed API
You can use a deployed version of ValueNet via https://valuenet.cloudlab.zhaw.ch. 
This API is working with a deployed postgres database which can be found in [hack_zurich_database.dmp](data/hack_zurich/hack_zurich_database.dmp). 

You might play around with the visual client (use the speech to text button) or access the API directly. Make sure to choose the correct database (`hack_zurich`) and the API key `sjNmaCtviYzXWlS`.

Here an example how to use the API wit cURL:

```
curl 'https://valuenet.cloudlab.zhaw.ch/api/question/hack_zurich' \
  -X 'PUT' \
  -H 'X-API-Key: sjNmaCtviYzXWlS' \
  -H 'Content-Type: application/json;charset=UTF-8' \
  --data-raw '{"question":"What is the share of electric cars in 2016 for Wetzikon?", "beam_size": 2}' 
```
You will receive a JSON response of the following format, where the actual result is found in `result` and the executed query in `sql`. 
The array `beams`contains more solution candidates (ordered by the score), but might not be of interest to you.

```json
{
  "beams": [
    [
      "SELECT T1.share_electric_cars FROM share_electric_cars AS T1 JOIN spatialunit AS T2 ON T1.spatialunit_id = T2.spatialunit_id WHERE T2.name = 'Wetzikon' and T1.year = 2016       ",
      8.053660922580296
    ],
    [
      "SELECT T1.share_electric_cars FROM share_electric_cars AS T1 JOIN spatialunit AS T2 ON T1.spatialunit_id = T2.spatialunit_id WHERE T1.year = 2016 and T2.name = 'Wetzikon'       ",
      2.968265109592014
    ]
  ],
  "potential_values_found_in_db": [
    "Wetzikon",
    "2016"
  ],
  "question": "What is the share of electric cars in 2016 for Wetzikon?",
  "question_tokenized": [
    "What",
    "is",
    "the",
    "share",
    "of",
    "electric",
    "cars",
    "in",
    "2016",
    "for",
    "Wetzikon",
    "?"
  ],
  "result": [
    [
      "0.3"
    ]
  ],
  "sem_ql": "Root1(3) Root(3) Sel(0) N(0) A(0) C(28) T(11) Filter(0) Filter(2) A(0) C(12) T(1) V(0) Filter(2) A(0) C(18) T(11) V(1)",
  "sql": "SELECT T1.share_electric_cars FROM share_electric_cars AS T1 JOIN spatialunit AS T2 ON T1.spatialunit_id = T2.spatialunit_id WHERE T2.name = 'Wetzikon' and T1.year = 2016       "
}
```


## Local Inference
To run the trained ValueNet model locally, you have two options:
1. Pull or build the inference docker image and run it, connected to your local database _(easy)_. 
2. Run the inference API ([manual_inference_api.py](src/manual_inference/manual_inference_api.py)) from your terminal / IDE, connected to your local database _(medium)_.

In both case you need to point the system to a database which contains your data. The easiest way is to install PostgreSQL locally and restore the database dump [hack_zurich_database.dmp](data/hack_zurich/hack_zurich_database.dmp), which contains all necessary data, tables, views and indices.

In case you plan to manipulate the database schema, make sure to also adapt the schema-file which is used by ValueNet at inference time ([tables.json](data/hack_zurich/original/tables.json)). This file contains a high level schema of the database (some tables might be abstracted by simple views) and is the foundation on which ValueNet is synthesizing a query.

After you adapted the schema file, make sure to re-build your docker image as the schema file is built in (see [build_inference_docker.sh](docker/build_inference_docker.sh))

### Run inference docker image (pre-built or adapted by you)
The docker image used in this case is [Dockerfile](docker/inference/Dockerfile). You can pull the pre-built image from https://hub.docker.com/repository/docker/ursinbrunner/valuenet-inference-hack-zurich.

```
docker pull ursinbrunner/valuenet-inference-hack-zurich:1.0
```

To run the docker image make sure a **NVIDIA GPU** is available (either on your notebook or in the cloud you prefer) and your Docker environment supports GPUs. To do so, you might follow the official nvidia guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html

To run the docker image use the following command. You might wanna change the **database connection** to point to your database. To do so override the environment variables, as seen here for the database host/password and API key.

```
docker run --gpus all -e API_KEY=api_key_you_plan_to_use -e DB_USER=postgres -e DB_PW=your_secret_db_password -e DB_HOST=localhost -e DB_PORT=5432 -e NER_API_SECRET=your_google_ner_api_key -p 5000:5000 --network="host" ursinbrunner/valuenet-inference-hack-zurich:1.0
```
The parameter `--network="host"` is only necessary if you run the database on the docker host system.

#### Adapt docker image
You might have to adapt the inference docker image, if for example you adapt the database schema file. To do so, have a look at the docker file itself [Dockerfile](docker/inference/Dockerfile) and use/adapt the build script ([build_inference_docker.sh](docker/build_inference_docker.sh))

### Run inference API locally

In case you plan to modify/re-train the model, might as well setup the project environment and run the inference locally via [manual_inference_api.py](src/manual_inference/manual_inference_api.py) script. 

To do so, follow the environment setup described in [Environment Setup](#environment-setup) and run [manual_inference_api.py](src/manual_inference/manual_inference_api.py). Specify command line parameters as seen in the inference [Dockerfile](docker/inference/Dockerfile) to point to your database.


## Train Model

TODO

### Environment Setup
TODO
