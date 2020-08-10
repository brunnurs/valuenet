import psycopg2
import torch

from intermediate_representation.sem_utils import alter_column0
from named_entity_recognition.api_ner.google_api_repository import remote_named_entity_recognition
from named_entity_recognition.pre_process_ner_values import pre_process, match_values_in_database
from spider.example_builder import build_example
from intermediate_representation.sem2sql.sem2SQL import transform


def _inference_semql(data_row, schemas, model):
    example = build_example(data_row, schemas)

    with torch.no_grad():
        results_all = model.parse(example, beam_size=1)
    results = results_all[0]
    # here we set assemble the predicted actions (including leaf-nodes) as string
    full_prediction = " ".join([str(x) for x in results[0].actions])

    prediction = example.sql_json['pre_sql']
    prediction['model_result'] = full_prediction

    return prediction, example


def _tokenize_question(tokenizer, question):
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions

    question_tokenized = tokenizer(question)

    return [str(token) for token in question_tokenized]


def _pre_process_values(row, schema_path, connection_config):
    ner_results = remote_named_entity_recognition(row['question'])
    row['ner_extracted_values'] = ner_results['entities']

    extracted_values = pre_process(row)

    row['values'] = match_values_in_database(row['db_id'], extracted_values, schema_path, connection_config)

    return row


def _semql_to_sql(prediction, schemas):
    alter_column0([prediction])
    result = transform(prediction, schemas[prediction['db_id']])
    return result[0]


def _execute_query(sql, connection_config):
    db_options = f"-c search_path={connection_config['database_schema']},public"

    conn = psycopg2.connect(database=connection_config['database'], user=connection_config['database_user'],
                            password=connection_config['database_password'], host=connection_config['database_host'],
                            port=connection_config['database_port'], options=db_options)
    cursor = conn.cursor()

    cursor.execute(sql)
    result = cursor.fetchall()

    conn.close()

    return result
