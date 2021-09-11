cp -R ../src inference
cp -R ../experiments/bart_best_model inference/models

mkdir -p inference/data/spider
cp -R ../data/spider/original inference/data/spider

mkdir -p inference/data/cordis
cp -R ../data/cordis/original inference/data/cordis

mkdir -p inference/data/hack_zurich
cp -R ../data/hack_zurich/original inference/data/hack_zurich

cd inference || exit

docker build -t ursinbrunner/valuenet-inference-hack-zurich:1.0 .
docker push ursinbrunner/valuenet-inference-hack-zurich:1.0

# remove everything except the Dockerfile
rm -R $(ls -I "Dockerfile" )