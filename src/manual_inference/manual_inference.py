import os
import pickle
import sqlite3
from pprint import pprint

import torch

from config import read_arguments_manual_inference
from intermediate_representation import semQL
from intermediate_representation.sem2sql.sem2SQL import transform
from intermediate_representation.sem_utils import alter_column0
from model.model import IRNet
from named_entity_recognition.api_ner.google_api_repository import remote_named_entity_recognition
from named_entity_recognition.pre_process_ner_values import pre_process, match_values_in_database
from preprocessing.process_data import process_datas
from preprocessing.utils import merge_data_with_schema
from spider import spider_utils
from spider.example_builder import build_example
from utils import setup_device, set_seed_everywhere

from spacy.lang.en import English

from termcolor import colored


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


def _pre_process_values(row):
    ner_results = remote_named_entity_recognition(row['question'])
    row['ner_extracted_values'] = ner_results['entities']

    extracted_values = pre_process(row)

    row['values'] = match_values_in_database(row['db_id'], extracted_values)

    return row


def _semql_to_sql(prediction, schemas):
    alter_column0([prediction])
    result = transform(prediction, schemas[prediction['db_id']])
    return result[0]


def _execute_query(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    cursor.execute(sql)
    result = cursor.fetchall()

    conn.close()

    return result


def _print_banner():
    print(colored('''
                                                                                                                                                                   
                                                                                                                                                           
VVVVVVVV           VVVVVVVV               lllllll                                       NNNNNNNN        NNNNNNNN                             tttt          
V::::::V           V::::::V               l:::::l                                       N:::::::N       N::::::N                          ttt:::t          
V::::::V           V::::::V               l:::::l                                       N::::::::N      N::::::N                          t:::::t          
V::::::V           V::::::V               l:::::l                                       N:::::::::N     N::::::N                          t:::::t          
 V:::::V           V:::::Vaaaaaaaaaaaaa    l::::l uuuuuu    uuuuuu      eeeeeeeeeeee    N::::::::::N    N::::::N    eeeeeeeeeeee    ttttttt:::::ttttttt    
  V:::::V         V:::::V a::::::::::::a   l::::l u::::u    u::::u    ee::::::::::::ee  N:::::::::::N   N::::::N  ee::::::::::::ee  t:::::::::::::::::t    
   V:::::V       V:::::V  aaaaaaaaa:::::a  l::::l u::::u    u::::u   e::::::eeeee:::::eeN:::::::N::::N  N::::::N e::::::eeeee:::::eet:::::::::::::::::t    
    V:::::V     V:::::V            a::::a  l::::l u::::u    u::::u  e::::::e     e:::::eN::::::N N::::N N::::::Ne::::::e     e:::::etttttt:::::::tttttt    
     V:::::V   V:::::V      aaaaaaa:::::a  l::::l u::::u    u::::u  e:::::::eeeee::::::eN::::::N  N::::N:::::::Ne:::::::eeeee::::::e      t:::::t          
      V:::::V V:::::V     aa::::::::::::a  l::::l u::::u    u::::u  e:::::::::::::::::e N::::::N   N:::::::::::Ne:::::::::::::::::e       t:::::t          
       V:::::V:::::V     a::::aaaa::::::a  l::::l u::::u    u::::u  e::::::eeeeeeeeeee  N::::::N    N::::::::::Ne::::::eeeeeeeeeee        t:::::t          
        V:::::::::V     a::::a    a:::::a  l::::l u:::::uuuu:::::u  e:::::::e           N::::::N     N:::::::::Ne:::::::e                 t:::::t    tttttt
         V:::::::V      a::::a    a:::::a l::::::lu:::::::::::::::uue::::::::e          N::::::N      N::::::::Ne::::::::e                t::::::tttt:::::t
          V:::::V       a:::::aaaa::::::a l::::::l u:::::::::::::::u e::::::::eeeeeeee  N::::::N       N:::::::N e::::::::eeeeeeee        tt::::::::::::::t
           V:::V         a::::::::::aa:::al::::::l  uu::::::::uu:::u  ee:::::::::::::e  N::::::N        N::::::N  ee:::::::::::::e          tt:::::::::::tt
            VVV           aaaaaaaaaa  aaaallllllll    uuuuuuuu  uuuu    eeeeeeeeeeeeee  NNNNNNNN         NNNNNNN    eeeeeeeeeeeeee            ttttttttttt  
                                                                                                                                                           
                                                                                                                                                           
                                                                                                                                                           
                                                                                                                                                           
                                                                                                                                                           
                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                        
        ''', 'blue'))


if __name__ == '__main__':
    args = read_arguments_manual_inference()

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    schemas_raw, schemas_dict = spider_utils.load_schema(args.data_dir)

    grammar = semQL.Grammar()
    model = IRNet(args, device, grammar)
    model.to(device)

    # load the pre-trained parameters
    model.load_state_dict(torch.load(args.model_to_load))
    model.eval()
    print("Load pre-trained model from '{}'".format(args.model_to_load))

    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    with open(os.path.join(args.conceptNet, 'english_RelatedTo.pkl'), 'rb') as f:
        related_to_concept = pickle.load(f)

    with open(os.path.join(args.conceptNet, 'english_IsA.pkl'), 'rb') as f:
        is_a_concept = pickle.load(f)

    while True:

        _print_banner()
        question = input(colored(f"You are using the database '{args.database}'. Type your question:", 'green', attrs=['bold']))

        try:

            row = {
                'question': question,
                'query': 'DUMMY',
                'db_id': args.database,
                'question_toks': _tokenize_question(tokenizer, question)
            }

            print(colored(f"question has been tokenized to : { row['question_toks'] }", 'cyan', attrs=['bold']))

            data, table = merge_data_with_schema(schemas_raw, [row])

            pre_processed_data = process_datas(data, related_to_concept, is_a_concept)

            pre_processed_with_values = _pre_process_values(pre_processed_data[0])

            print(f"we found the following potential values in the question: {row['values']}")

            prediction, example = _inference_semql(pre_processed_with_values, schemas_dict, model)

            print(f"Results from schema linking (question token types): {example.src_sent}")
            print(f"Results from schema linking (column types): {example.col_hot_type}")

            print(colored(f"Predicted SemQL-Tree: {prediction['model_result']}", 'magenta', attrs=['bold']))
            print()
            sql = _semql_to_sql(prediction, schemas_dict)

            print(colored(f"Transformed to SQL: {sql}", 'cyan', attrs=['bold']))
            print()
            result = _execute_query(sql, args.database_path)

            print(f"Executed on the database '{args.database}'. Results: ")
            for row in result:
                print(colored(row, 'green'))

        except Exception as e:
            print("Exception: " + str(e))

        input(colored("Press [Enter] to continue with your next question.", 'red', attrs=['bold']))
