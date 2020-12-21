import os

import wandb
from pytictoc import TicToc

from config import read_arguments_evaluation
from intermediate_representation.sem2sql.sem2SQL import transform_semQL_to_sql
from spider import spider_utils
from spider.test_suite_eval.evaluation import evaluate

####### use this code to test the ground-truth directly with the Spider-Execution validation
####### that way we can test the SQL -> SemQL -> SQL round trip. Use sql_data for the whole training set or val_sql_data for dev

args = read_arguments_evaluation()

sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=False)

for d in val_sql_data:
    d['model_result'] = d['rule_label']

count_success, count_failed = transform_semQL_to_sql(val_table_data, val_sql_data, args.prediction_dir)

print("Transformed {} samples successful to SQL. {} samples failed. Generated the files a 'ground_truth.txt' "
      "and a 'output.txt' file. We now use the official Spider evaluation script to evaluate this files.".format(
    count_success, count_failed))

t = TicToc()
t.tic()
wandb.init(project="proton")

evaluate(os.path.join(args.prediction_dir, 'ground_truth.txt'),
         os.path.join(args.prediction_dir, 'output.txt'),
         os.path.join(args.data_dir, "testsuite_databases"),
         'exec', None, False, False, False, 1, quickmode=False)
t.toc()
