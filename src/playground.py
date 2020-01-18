import os

from src.spider.evaluation.spider_evaluation import spider_evaluation, build_foreign_key_map_from_json

kmaps = build_foreign_key_map_from_json("data/spider/tables.json")

spider_evaluation(os.path.join("experiments/exp__20200115_191836/ground_truth.txt"),
                  os.path.join("experiments/exp__20200115_191836/output.txt"),
                  os.path.join("data/spider/original/database"),
                  "match", kmaps)