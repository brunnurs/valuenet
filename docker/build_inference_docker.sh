cp -R ../src inference
cp -R ../experiments/bart_best_model inference/models

mkdir -p inference/data/spider
cp -R ../data/spider/original inference/data/spider

mkdir -p inference/data/cordis
cp -R ../data/cordis/original inference/data/cordis

mkdir -p inference/data/hack_zurich
cp -R ../data/hack_zurich/original inference/data/hack_zurich

mkdir -p inference/data/oncomx
cp -R ../data/oncomx/original inference/data/oncomx

cd inference || exit

docker build -t ursinbrunner/proton-inference:3.4 .
docker push ursinbrunner/proton-inference:3.4

# remove everything except the Dockerfile
rm -R $(ls -I "Dockerfile" )