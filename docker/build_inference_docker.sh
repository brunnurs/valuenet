cp -R ../src inference
cp -R ../experiments/ner_values_best inference/models
cp -R ../data inference

rm -R inference/data/spider

cd inference || exit

docker build -t ursinbrunner/proton-inference:latest .

# remove everything except the Dockerfile
rm -R $(ls -I "Dockerfile" )