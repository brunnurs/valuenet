import os

import torch
from tqdm import tqdm

from src.spider.example_builder import build_example


def get_key(value, dic):
    for key in dic:
        if dic[key] is value:
            return key


def evaluate(model, dev_loader, table_data, beam_size):
    sketch_correct, rule_label_correct, total = 0, 0, 0

    for batch in tqdm(dev_loader, desc="Evaluating"):
        model.eval()

        for data_row in batch:
            example = build_example(data_row, table_data)

            with torch.no_grad():
                results_all = model.parse(example, beam_size=beam_size)

            results = results_all[0]
            list_preds = []
            try:

                pred = " ".join([str(x) for x in results[0].actions])
                for x in results:
                    list_preds.append(" ".join(str(x.actions)))
            except Exception as e:
                # print('Epoch Acc: ', e)
                # print(results)
                # print(results_all)
                pred = ""

            simple_json = example.sql_json['pre_sql']

            simple_json['sketch_result'] = " ".join(str(x) for x in results_all[1])
            simple_json['model_result'] = pred

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.tgt_actions])

            if truth_sketch == simple_json['sketch_result']:
                sketch_correct += 1
            if truth_rule_label == simple_json['model_result']:
                rule_label_correct += 1
            total += 1

    return float(sketch_correct) / float(total), float(rule_label_correct) / float(total)
