FROM nvcr.io/nvidia/pytorch:19.12-py3

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
RUN pip install "textdistance[extras]"
RUN pip install spacy
RUN pip install psycopg2-binary
RUN pip install flask
RUN pip install flask-cors

COPY src /workspace/src
COPY data /workspace/data
COPY models /workspace/models

RUN python src/tools/download_nltk.py

ENV PYTHONPATH /workspace/src

WORKDIR /workspace

ENTRYPOINT python src/manual_inference/manual_inference_api.py --api_key=$API_KEY --model_to_load=models/best_model.pt --database_host=testbed.inode.igd.fraunhofer.de --database_port=18001 --database_user=postgres --database_password=vdS83DJSQz2xQ --database_schema=unics_cordis --conceptNet=data/cordis/conceptNet
