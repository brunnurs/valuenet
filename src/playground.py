import json

# with open('data/spider/preprocessed_with_values.json', 'r', encoding='utf-8') as json_file:
#     data = json.load(json_file)
#     for row in data:
#         values = row['values']
#         if values:
#             candidates = row['ner_extracted_values_processed']
#             print(f'Values: {values}          Candiates: {candidates}')

from helpers.get_values_from_sql import format_groundtruth_value

with open('data/spider/train.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

count = 0
for row in data:
    if len(row['values']) >= 3:
        print(row['values'])
        print(row['ner_extracted_values_processed'])
        print(row['rule_label'])
        print(row['question'])
        print(row['query'])
        print()
        print()

        count += 1

print(count)