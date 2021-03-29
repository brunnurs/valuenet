import json
from pathlib import Path

# from spider.spider_utils import load_schema
#
# _, schema_dict = load_schema('data/cordis/original/tables.json')
# print(schema_dict)

with open(Path('data/spider/original/tables.json'), 'r', encoding='utf-8') as json_file:
    train_data_spider = json.load(json_file)

with open(Path('data/cordis/original/tables.json'), 'r', encoding='utf-8') as json_file:
    train_data_cordis = json.load(json_file)

with open(Path('data/spider/original/tables_spider_cordis.json'), 'w') as f:
    json.dump(train_data_spider + train_data_cordis, f, indent=2)
