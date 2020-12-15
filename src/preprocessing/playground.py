import json
from pathlib import Path

with open(Path('data/spider/original/train_spider.json'), 'r', encoding='utf-8') as json_file:
    train_data = json.load(json_file)

with open(Path('data/spider/original/train_others.json'), 'r', encoding='utf-8') as json_file:
    train_data_others = json.load(json_file)

with open(Path('data/spider/original/train.json'), 'w') as f:
    json.dump(train_data + train_data_others, f, indent=4)
