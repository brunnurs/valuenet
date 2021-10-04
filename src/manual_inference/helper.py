import copy
import sqlite3
from pathlib import Path

import psycopg2
import torch

from config import Config
from intermediate_representation.sem_utils import alter_column0
from named_entity_recognition.api_ner.google_api_repository import remote_named_entity_recognition
from preprocessing.pre_process import pre_process
from spider import spider_utils
from spider.example_builder import build_example
from intermediate_representation.sem2sql.sem2SQL import transform


def _inference_semql(data_row, schemas, model, beam_size):
    original_row = copy.deepcopy(data_row)
    example = build_example(data_row, schemas)

    with torch.no_grad():
        results = model.parse(example, beam_size=beam_size)[0]

    # here we set assemble the predicted actions (including leaf-nodes) as string.
    full_prediction = " ".join([str(x) for x in results[0].actions])

    all_beams = []
    # in case someone is requesting more beams, we assemble them here as well.
    for beam in results:
        # we normalize the score by dividing through the number of steps this beam has
        all_beams.append((" ".join([str(x) for x in beam.actions]), beam.score.item() / beam.t))

    original_row['model_result'] = full_prediction

    return original_row, example, all_beams


def tokenize_question(tokenizer, question):
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions

    question_tokenized = tokenizer(question)

    return [str(token) for token in question_tokenized]


def _pre_processing(example, db_value_finder, ner_api_secret):
    ner_information = remote_named_entity_recognition(example['question'], ner_api_secret)

    token_grouped, token_types, column_matches, value_candidates, _ = pre_process(0, example, ner_information, db_value_finder, is_training=False)

    example['question_arg'] = token_grouped
    example['question_arg_type'] = token_types
    example['column_matches'] = column_matches
    example['values'] = value_candidates

    return example


def _semql_to_sql(prediction, schemas):
    alter_column0([prediction])
    result = transform(prediction, schemas[prediction['db_id']])
    return result[0]


def _execute_query_postgresql(sql, database, connection_config):
    db_options = f"-c search_path={connection_config['database_schema']},public"

    conn = psycopg2.connect(database=database, user=connection_config['database_user'],
                            password=connection_config['database_password'], host=connection_config['database_host'],
                            port=connection_config['database_port'], options=db_options)
    cursor = conn.cursor()

    cursor.execute(sql)
    result = cursor.fetchall()

    conn.close()

    return result


def _execute_query_sqlite(sql, database_path, db):
    full_database_path = Path(database_path) / db / f'{db}.sqlite'

    conn = sqlite3.connect(str(full_database_path))
    cursor = conn.cursor()

    cursor.execute(sql)
    result = cursor.fetchall()

    conn.close()

    return result


def get_schemas_spider():
    base_path = Path(Config.DATA_PREFIX) / 'spider' / 'original'
    schema_path = str(base_path / 'tables.json')
    database_path = str(base_path / 'database')

    schemas_raw, schemas_dict = spider_utils.load_schema(schema_path)

    return schemas_raw, schemas_dict, schema_path, database_path


def get_schemas_cordis():
    base_path = Path(Config.DATA_PREFIX) / 'cordis' / 'original'
    schema_path = str(base_path / 'tables.json')
    database_path = str(base_path / 'database')

    schemas_raw, schemas_dict = spider_utils.load_schema(schema_path)

    return schemas_raw, schemas_dict, schema_path, database_path


def get_schema_hack_zurich():
    base_path = Path(Config.DATA_PREFIX) / 'hack_zurich' / 'original'
    schema_path = str(base_path / 'tables.json')

    schemas_raw, schemas_dict = spider_utils.load_schema(schema_path)

    return schemas_raw, schemas_dict, schema_path


def get_schema_oncomx():
    base_path = Path(Config.DATA_PREFIX) / 'oncomx' / 'original'
    schema_path = str(base_path / 'tables.json')

    schemas_raw, schemas_dict = spider_utils.load_schema(schema_path)

    return schemas_raw, schemas_dict, schema_path



def get_data_folder_by_database(database_name):
    if database_name == 'cordis_temporary' or database_name == 'cordis':
        return 'cordis'
    if database_name == 'hack_zurich':
        return 'hack_zurich'
    if database_name == 'oncomx_v1_0_25_small':
        return 'oncomx'
    else:
        return 'spider'

