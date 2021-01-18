cp -R ../src inference
cp -R ../experiments/bart_best_model inference/models
cp -R ../data inference

cd inference || exit

docker build -t ursinbrunner/proton-inference:2.0 .
docker push ursinbrunner/proton-inference:2.0

# remove everything except the Dockerfile
rm -R $(ls -I "Dockerfile" )