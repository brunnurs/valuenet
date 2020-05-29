#### This script is needed to convert exported runs from Weights & Biases into the LaTex plot format ###########
import csv

# with open('/home/ursin/Downloads/ValueNet_light.csv', "r", encoding='utf-8') as f:
with open('/home/ursin/Downloads/ValueNet.csv', "r", encoding='utf-8') as f:
# with open('/home/ursin/Downloads/ExactMatchingAccuracy.csv', "r", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=",")
    # skip header
    next(reader)
    for line in reader:
        epoch = line[0]
        accuracy_adapted = float(line[1]) * 100
        print(f'({epoch}, {accuracy_adapted})')

