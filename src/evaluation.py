import copy
import os
from pathlib import Path

import torch
import wandb
from tqdm import tqdm
import json

from config import read_arguments_evaluation
from data_loader import get_data_loader
from intermediate_representation import semQL
from intermediate_representation.sem2sql.sem2SQL import transform_semQL_to_sql
from manual_inference.helper import _execute_query_postgresql
from model.model import IRNet
from spider import spider_utils
from spider.example_builder import build_example
from spider.test_suite_eval.exec_eval import result_eq
from utils import setup_device, set_seed_everywhere
import spider.test_suite_eval.evaluation as spider_evaluation


def evaluate(model, dev_loader, schema, beam_size):
    model.eval()

    sketch_correct, rule_label_correct, found_in_beams, not_all_values_found, total = 0, 0, 0, 0, 0
    predictions = []
    for batch in tqdm(dev_loader, desc="Evaluating"):

        for data_row in batch:
            original_row = copy.deepcopy(data_row)

            try:
                example = build_example(data_row, schema)
            except Exception as e:
                print("Exception while building example (evaluation): {}".format(e))
                continue

            with torch.no_grad():
                results_all = model.parse(example, beam_size=beam_size)

            results = results_all[0]
            all_predictions = []
            try:
                # here we set assemble the predicted actions (including leaf-nodes) as string
                full_prediction = " ".join([str(x) for x in results[0].actions])
                for beam in results:
                    all_predictions.append(" ".join([str(x) for x in beam.actions]))
            except Exception as e:
                # print(e)
                full_prediction = ""

            prediction = original_row

            # here we set assemble the predicted sketch actions as string
            prediction['sketch_result'] = " ".join(str(x) for x in results_all[1])
            prediction['model_result'] = full_prediction

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.semql_actions])

            if prediction['all_values_found']:
                if truth_sketch == prediction['sketch_result']:
                    sketch_correct += 1
                if truth_rule_label == prediction['model_result']:
                    rule_label_correct += 1
                elif truth_rule_label in all_predictions:
                    found_in_beams += 1
            else:
                question = prediction['question']
                tqdm.write(f'Not all values found during pre-processing for question "{question}". Replace values with dummy to make query fail')
                prediction['values'] = [1] * len(prediction['values'])
                not_all_values_found += 1

            total += 1

            predictions.append(prediction)

    print(f"in {found_in_beams} times we found the correct results in another beam (failing queries: {total - rule_label_correct})")

    return float(sketch_correct) / float(total), float(rule_label_correct) / float(total), float(not_all_values_found) / float(total), predictions


def transform_to_sql_and_evaluate_with_spider(predictions, table_data, experiment_dir, data_dir, training_step):
    total_count, failure_count = transform_semQL_to_sql(table_data, predictions, experiment_dir)

    spider_eval_results = spider_evaluation.evaluate(os.path.join(experiment_dir, 'ground_truth.txt'),
                                                     os.path.join(experiment_dir, 'output.txt'),
                                                     os.path.join(data_dir, "testsuite_databases"),
                                                     "exec",
                                                     None,
                                                     False,
                                                     False,
                                                     False,
                                                     training_step,
                                                     quickmode=False)

    return total_count, failure_count, spider_eval_results


def _remove_unnecessary_distinct(p, g):
    p_tokens = p.split(' ')
    g_tokens = g.split(' ')

    if 'distinct' not in g_tokens and 'DISTINCT' not in g_tokens:
        return ' '.join([p_token for p_token in p_tokens if p_token != 'DISTINCT'])
    else:
        return p


def evaluate_cordis(groundtruth_path: Path, prediction_path: Path, database: str, connection_config: dict, do_not_verify_distinct=True):

    with open(groundtruth_path, 'r', encoding='utf-8') as f:
        groundtruth_full_lines = f.readlines()
        groundtruth = [g.strip().split('\t')[0] for g in groundtruth_full_lines]
        questions = [g.strip().split('\t')[2] for g in groundtruth_full_lines]

    with open(prediction_path, 'r', encoding='utf-8') as f:
        prediction = f.readlines()

    n_success = 0
    for p, g, q in zip(prediction, groundtruth, questions):

        if do_not_verify_distinct:
            p = _remove_unnecessary_distinct(p, g)

        try:
            results_prediction = _execute_query_postgresql(p, database, connection_config)
        except Exception as ex:
            print("Could not execute query. Error:", ex)
            print(f"Q: {q.strip()}")
            print(f"P: {p.strip()}")
            print(f"G: {g.strip()}")
            print()

            continue

        results_groundtruth = _execute_query_postgresql(g, database, connection_config)

        if result_eq(results_groundtruth, results_prediction, order_matters=True):
            n_success += 1
        else:
            print("Results not equal for:")
            print(f"Q: {q.strip()}")
            print(f"P: {p.strip()}")
            print(f"G: {g.strip()}")
            print()

    print(f"================== Total score: {n_success} of {len(groundtruth)} ({100 / len(groundtruth) * n_success}%) ==================")


def main():
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

    print("Predicted {} examples. Start now converting them to SQL. Sketch-Accuracy: {}, Accuracy: {}, Not all values found: {}".format(
            len(dev_loader), sketch_acc, acc, not_all_values_found))

    with open(os.path.join(args.prediction_dir, 'predictions_sem_ql.json'), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    # with open(os.path.join(args.prediction_dir, 'predictions_sem_ql.json'), 'r', encoding='utf-8') as json_file:
    #     predictions = json.load(json_file)

    count_success, count_failed = transform_semQL_to_sql(val_table_data, predictions, args.prediction_dir)

    print("Transformed {} samples successful to SQL. {} samples failed. Generated the files a 'ground_truth.txt' "
          "and a 'output.txt' file.".format(
        count_success, count_failed))

    if args.evaluation_type == 'spider':

        print('We now use the official Spider evaluation script to evaluate the generated/ground truth files.')
        wandb.init(project="proton")

        spider_evaluation.evaluate(os.path.join(args.prediction_dir, 'ground_truth.txt'),
                                   os.path.join(args.prediction_dir, 'output.txt'),
                                   os.path.join(args.data_dir, "testsuite_databases"),
                                   'exec', None, False, False, False, 1, quickmode=False)

    elif args.evaluation_type == 'cordis':

        connection_config = {k: v for k, v in vars(args).items() if k.startswith('database')}

        evaluate_cordis(Path(args.prediction_dir) / 'ground_truth.txt',
                        Path(args.prediction_dir) / 'output.txt',
                        args.database,
                        connection_config)
    else:
        raise NotImplemented('Only Spider and CORDIS evaluation are implemented so far.')


if __name__ == '__main__':
    main()
