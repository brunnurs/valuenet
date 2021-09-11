FROM nvcr.io/nvidia/pytorch:20.12-py3

ENTRYPOINT bash

RUN apt update
RUN apt-get -y install default-libmysqlclient-dev

# Don't just use the requirements file and install torch etc.... the nvidia image already
# comes with a pre-installed torch etc, so installing another torch/pytorch (and also the transitive torch dependency of transformers for example)
# will cause quite a bit of trouble.
RUN pip install nltk
RUN pip install tqdm
RUN pip install pattern
RUN pip install transformers
RUN pip install pytictoc
RUN pip install wandb
RUN pip install pyyaml
RUN pip install word2number
RUN pip install sqlparse
RUN pip install "textdistance[extras]"
RUN pip install spacy
# in contrary to the pipfile we need some binaries for postgres too.
RUN pip install psycopg2-binary
RUN pip install flask
RUN pip install flask-cors

COPY src /workspace/src
COPY data /workspace/data
COPY models /workspace/models

RUN python src/tools/download_nltk.py
RUN python -m spacy download en_core_web_sm

ENV PYTHONPATH /workspace/src

WORKDIR /workspace

ENV API_KEY 1234
ENV MODEL_TO_LOAD models/best_model.pt
ENV DB_HOST localhost
ENV DB_PORT 5432
ENV DB_USER postgres
ENV DB_PW dummy
ENV DB_SCHEMA public
ENV NER_API_SECRET PLEASE_ADD_YOUR_OWN_GOOGLE_API_KEY_HERE

ENTRYPOINT python src/manual_inference/manual_inference_api.py --api_key=$API_KEY --model_to_load=$MODEL_TO_LOAD --ner_api_secret=$NER_API_SECRET --database_host=$DB_HOST --database_port=$DB_PORT --database_user=$DB_USER --database_password=$DB_PW --database_schema=$DB_SCHEMA
