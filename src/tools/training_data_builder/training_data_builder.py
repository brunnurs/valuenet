import argparse
import copy
import json
from pathlib import Path

from spacy.lang.en import English

from manual_inference.helper import get_schemas_cordis, tokenize_question
from spider.test_suite_eval.process_sql import get_sql
from tools.training_data_builder.schema import build_schema_mapping, SchemaIndex

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def transform_sample(sample, schema_dict):
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


def main():
    # load all schemas necessary for your training data. Right now cordis is enough.
    _, schemas_dict, _, _ = get_schemas_cordis()

    # There can be multiple files with training data which we will concatenate.
    # the training data needs to be an array of object each having the following properties:
    # 'db_id' --> name of the database
    # 'question' --> natural language question
    # 'query' --> SQL query as one string
    samples = []

    path_training_samples_1: Path = Path('data/cordis/trees/all_adapted.json')
    path_training_samples_2: Path = Path('data/cordis/handmade_training_data/handmade_data_train.json')

    for path in [path_training_samples_1, path_training_samples_2]:
        with open(path, 'r', encoding='utf-8') as file_handle:
            data = json.load(file_handle)

            for sample in data:
                transformed = transform_sample(sample, schemas_dict)
                samples.append(transformed)

    with open(Path('data/cordis/original/train.json'), 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    # now lets do the same for DEV, assuming there is only handmade dev data.
    samples = []

    path_dev_samples: Path = Path('data/cordis/handmade_training_data/handmade_data_dev.json')
    with open(path_dev_samples, 'r', encoding='utf-8') as file_handle:
        data = json.load(file_handle)

        for sample in data:
            transformed = transform_sample(sample, schemas_dict)
            samples.append(transformed)

    with open(Path('data/cordis/original/dev.json'), 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)


if __name__ == '__main__':
    main()
