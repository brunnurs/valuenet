cp src/download_nltk.py docker
pipenv lock --requirements > docker/requirements.txt
docker build -t ursinbrunner/proton:latest docker