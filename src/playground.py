import json

with open('data/spider/dev.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

ner_values_only = []
for row in data:
    ner_extracted_values = row['ner_extracted_values']
    ner_extracted_values['question'] = row['question']
    ner_values_only.append(ner_extracted_values)

with open('data/spider/ner_dev.json', 'w', encoding='utf-8') as f:
    json.dump(ner_values_only, f, indent=2)
