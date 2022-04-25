import argparse
import json
import os

from preprocessing.sql2SemQL import Parser
from preprocessing.utils import load_dataSets


def validate_values_contained_in_ner(ner_entities, values_ground_truth):
    for value_gt in values_ground_truth:
        if not any(ner_entity['name'].lower() == value_gt.lower() for ner_entity in ner_entities):
            print(f'NER could not find value {value_gt} we were looking for. We might be able to find it later based on heuristics (see pre_processing.py).')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--ner_path', type=str, help='file containing the values extracted by NER (e.g. ner_train.json)', required=True)
    args = arg_parser.parse_args()

    # loading dataSets
    data, table = load_dataSets(args)
    processed_data = []

    with open(os.path.join(args.ner_path), 'r', encoding='utf-8') as json_file:
        ner_data = json.load(json_file)

    if len(ner_data) != len(data):
        raise ValueError(f'There are {len(ner_data)} NER data rows and {len(data)} samples in the data set. Something is wrong! '
                         f'Comment in the next few lines to figure out which questions are missing (e.g. failed during the NER step)')

    # for i in range(len(data)):
    #     if data[i]['question'] != ner_data[i]['question']:
    #         print(f'Question {i} is not the same between the two files')

    for row, ner_row in zip(data, ner_data):

        if len(row['sql']['select'][1]) > 5:
            ner_row['values'] = ['Cant handle this sample as it has more than 5 select columns']
            continue

        parser = Parser(build_value_list=True)
        _ = parser.full_parse(row)

        ner_row['values'] = parser.values
        # validate_values_contained_in_ner(ner_row['entities'], values_formatted)

        question = row['question']

        print(f'Found values {parser.values} for question: "{question}"')

    print(f'Read out values from {len(data)} questions and added it to NER-file {args.ner_path}')
    with open(args.ner_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(ner_data, indent=2))
