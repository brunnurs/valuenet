import json
from pathlib import Path

from spider.spider_utils import load_schema

_, schema_dict = load_schema('data/cordis/original/tables.json')
print(schema_dict)

# with open(Path('data/spider/original/train_spider.json'), 'r', encoding='utf-8') as json_file:
#     train_data = json.load(json_file)
#
# with open(Path('data/spider/original/train_others.json'), 'r', encoding='utf-8') as json_file:
#     train_data_others = json.load(json_file)
#
# with open(Path('data/spider/original/train.json'), 'w') as f:
#     json.dump(train_data + train_data_others, f, indent=4)
