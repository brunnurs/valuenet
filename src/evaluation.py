import os

import torch
from tqdm import tqdm
import json

from config import read_arguments_evaluation
from data_loader import get_data_loader
from intermediate_representation import semQL
from intermediate_representation.sem2sql.sem2SQL import transform_semQL_to_sql
from model.model import IRNet
from spider import spider_utils
from spider.evaluation.spider_evaluation import spider_evaluation, build_foreign_key_map_from_json
from spider.example_builder import build_example
from utils import setup_device, set_seed_everywhere


def evaluate(model, dev_loader, table_data, beam_size):
    model.eval()

    sketch_correct, rule_label_correct, not_all_values_found, total = 0, 0, 0, 0
    predictions = []
    for batch in tqdm(dev_loader, desc="Evaluating"):

        for data_row in batch:
            try:
                example = build_example(data_row, table_data)
            except Exception as e:
                print("Exception while building example (evaluation): {}".format(e))
                continue

            with torch.no_grad():
                results_all = model.parse(example, beam_size=beam_size)

            results = results_all[0]
            list_preds = []
            try:
                # here we set assemble the predicted actions (including leaf-nodes) as string
                full_prediction = " ".join([str(x) for x in results[0].actions])
                for x in results:
                    list_preds.append(" ".join(str(x.actions)))
            except Exception as e:
                # print(e)
                full_prediction = ""

            prediction = example.sql_json['pre_sql']

            # here we set assemble the predicted sketch actions as string
            prediction['sketch_result'] = " ".join(str(x) for x in results_all[1])
            prediction['model_result'] = full_prediction

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.tgt_actions])

            if prediction['all_values_found']:
                if truth_sketch == prediction['sketch_result']:
                    sketch_correct += 1
                if truth_rule_label == prediction['model_result']:
                    rule_label_correct += 1
            else:
                question = prediction['question']
                print(f'Not all values found during pre-processing for question {question}. Replace values with dummy to make query fail')
                prediction['values'] = [1] * len(prediction['values'])
                not_all_values_found += 1

            total += 1

            predictions.append(prediction)

    return float(sketch_correct) / float(total), float(rule_label_correct) / float(total), float(not_all_values_found) / float(total), predictions


def transform_to_sql_and_evaluate_with_spider(predictions, table_data, data_dir, experiment_dir, tb_writer, training_step):
    total_count, failure_count = transform_semQL_to_sql(table_data, predictions, experiment_dir)

    kmaps = build_foreign_key_map_from_json(os.path.join(data_dir, 'original', 'tables.json'))

    spider_eval_results = spider_evaluation(os.path.join(experiment_dir, 'ground_truth.txt'),
                                            os.path.join(experiment_dir, 'output.txt'),
                                            os.path.join(data_dir, "original", "database"),
                                            "exec",
                                            kmaps,
                                            tb_writer,
                                            training_step, print_stdout=False)

    return total_count, failure_count, spider_eval_results


if __name__ == '__main__':
    args = read_arguments_evaluation()

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=False)
    _, dev_loader = get_data_loader(sql_data, val_sql_data, args.batch_size, True, False)

    grammar = semQL.Grammar()
    model = IRNet(args, device, grammar)
    model.to(device)

    # load the pre-trained parameters
    model.load_state_dict(torch.load(args.model_to_load))
    print("Load pre-trained model from '{}'".format(args.model_to_load))

    sketch_acc, acc, not_all_values_found, predictions = evaluate(model,
                                                                  dev_loader,
                                                                  table_data,
                                                                  args.beam_size)

    print("Predicted {} examples. Start now converting them to SQL. Sketch-Accuracy: {}, Accuracy: {}, Not all values found: {}".format(len(dev_loader), sketch_acc, acc, not_all_values_found))


    with open(os.path.join(args.prediction_dir, 'predictions_sem_ql.json'), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    # with open(os.path.join(args.prediction_dir, 'predictions_sem_ql.json'), 'r', encoding='utf-8') as json_file:
    #     predictions = json.load(json_file)

    count_success, count_failed = transform_semQL_to_sql(val_table_data, predictions, args.prediction_dir)

    print("Transformed {} samples successful to SQL. {} samples failed. Generated the files a 'ground_truth.txt' "
          "and a 'output.txt' file. We now use the official Spider evaluation script to evaluate this files.".format(
        count_success, count_failed))

    kmaps = build_foreign_key_map_from_json(os.path.join(args.data_dir, 'original', 'tables.json'))

    spider_evaluation(os.path.join(args.prediction_dir, 'ground_truth.txt'),
                      os.path.join(args.prediction_dir, 'output.txt'),
                      os.path.join(args.data_dir, 'original', 'database'),
                      "exec", kmaps)
