cp -R ../src inference
cp -R ../experiments/bart_best_model inference/models

mkdir -p inference/data/spider
cp -R ../data/spider/original inference/data/spider

mkdir -p inference/data/cordis
cp -R ../data/cordis/original inference/data/cordis

cd inference || exit

docker build -t ursinbrunner/proton-inference:3.2 .
docker push ursinbrunner/proton-inference:3.2

# remove everything except the Dockerfile
rm -R $(ls -I "Dockerfile" )