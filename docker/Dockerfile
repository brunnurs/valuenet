FROM nvcr.io/nvidia/pytorch:19.12-py3

COPY download_nltk.py /tmp

RUN apt update
RUN apt-get -y install default-libmysqlclient-dev


# the following lines are necessary to remote debug into the docker via SSH.
# make sure you also start the ssh service at the entrypoint!
EXPOSE 22

RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd

ADD ssh_keys.tar.gz /tmp/
RUN mkdir -p /root/.ssh \
    && mv /tmp/ssh_keys/authorized_keys /root/.ssh/authorized_keys \
    && chown root:root /root/.ssh/authorized_keys \
    && rm -rf /tmp/ssh_keys ssh_keys.tar.gz

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

RUN wandb login c9bac4494d41935972b1da58001870aec716e3bb

RUN python /tmp/download_nltk.py

ENV PYTHONPATH /workspace

WORKDIR /workspaces

ENTRYPOINT service ssh restart && bash
