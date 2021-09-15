import argparse
import copy
import json
from pathlib import Path
from typing import List

from spacy.lang.en import English

from manual_inference.helper import get_schemas_cordis, tokenize_question, get_schema_hack_zurich
from spider.test_suite_eval.process_sql import get_sql
from tools.training_data_builder.schema import build_schema_mapping, SchemaIndex

def transform_sample(sample, schema_dict, tokenizer):
    database = sample['db_id']
    query = sample['query']
    question = sample['question']

    schema_mapping = build_schema_mapping(schema_dict[database])
    schema = SchemaIndex(schema_mapping, schema_dict[database]['column_names_original'], schema_dict[database]['table_names_original'])

    spider_sql_structure, sql_tokenized = get_sql(schema, query)

    return {
        'db_id': database,
        'question': question,
        'question_toks': tokenize_question(tokenizer, question),
        'query': query,
        'sql': spider_sql_structure,
        'query_toks': sql_tokenized,
    }


def main(args: argparse.Namespace):
    nlp = English()

    # load schema necessary for your training data.
    if args.data == 'cordis':
        _, schemas_dict, _, _ = get_schemas_cordis()
    elif args.data == 'hack_zurich':
        _, schemas_dict, _ = get_schema_hack_zurich()
    else:
        raise ValueError('Dataset not yet supported')

    # There can be multiple files with training data which we will concatenate.
    # the training data needs to be an array of object each having the following properties:
    # 'db_id' --> name of the database
    # 'question' --> natural language question
    # 'query' --> SQL query as one string

    training_sample_paths: List[Path] = []

    training_sample_paths.append(Path(f'data/{args.data}/handmade_training_data/handmade_data_train.json'))

    if args.data == 'cordis':
        training_sample_paths.append(Path('data/cordis/trees/all_adapted.json'))

    samples = []
    for path in training_sample_paths:
        with open(path, 'r', encoding='utf-8') as file_handle:
            data = json.load(file_handle)

            for sample in data:
                transformed = transform_sample(sample, schemas_dict, nlp.tokenizer)
                samples.append(transformed)

    print(f'successfully transformed {len(samples)} samples for train split')

    with open(Path(f'data/{args.data}/original/train.json'), 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    # now lets do the same for DEV, assuming there is only handmade dev data.
    samples = []

    path_dev_samples: Path = Path(f'data/{args.data}/handmade_training_data/handmade_data_dev.json')
    with open(path_dev_samples, 'r', encoding='utf-8') as file_handle:
        data = json.load(file_handle)

        for sample in data:
            transformed = transform_sample(sample, schemas_dict, nlp.tokenizer)
            samples.append(transformed)

    print(f'successfully transformed {len(samples)} samples for dev split')

    with open(Path(f'data/{args.data}/original/dev.json'), 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data', type=str, choices=['cordis', 'hack_zurich'], required=True)

    args = arg_parser.parse_args()
    main(args)
