FROM nvcr.io/nvidia/pytorch:19.12-py3

COPY download_nltk.py /tmp

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

RUN wandb login c9bac4494d41935972b1da58001870aec716e3bb

RUN python /tmp/download_nltk.py

ENV PYTHONPATH /workspace

WORKDIR /workspace