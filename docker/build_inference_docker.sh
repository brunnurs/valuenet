cp -R ../src inference
cp -R ../experiments/bart_best_model inference/models
cp -R ../data/spider/original inference/data/spider
cp -R ../data/cordis/original inference/data/cordis

cd inference || exit

docker build -t ursinbrunner/proton-inference:3.0 .
docker push ursinbrunner/proton-inference:3.0

# remove everything except the Dockerfile
rm -R $(ls -I "Dockerfile" )