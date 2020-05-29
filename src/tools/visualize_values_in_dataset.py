import json
from functools import reduce

import matplotlib.pyplot as plt

#### We need this mainly to print the data four our paper ###########
with open('data/spider/preprocessed_with_values_train.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

    n_values = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for row in data:
        values = row['values']
        n_values[len(values)] += 1
    print(f'entries with values: {reduce(lambda memory, current: memory + current, n_values.values(), 0)}')
    print(f'Total number of values: {reduce(lambda memory, key: memory + key * n_values[key], n_values, 0)}')

    for key, value in n_values.items():
        print(f'({key},{value})')
