FROM python:3.7

COPY requirements.txt /tmp
COPY download_nltk.py /tmp

RUN pip install -r /tmp/requirements.txt

RUN python /tmp/download_nltk.py

ENV PYTHONPATH /workspace

WORKDIR /workspace