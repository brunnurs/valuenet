import os
import pickle
import torch

from config import read_arguments_manual_inference
from intermediate_representation import semQL
from manual_inference.helper import _tokenize_question, _pre_process_values, _inference_semql, _semql_to_sql, \
    _execute_query
from model.model import IRNet
from preprocessing.process_data import process_datas
from preprocessing.utils import merge_data_with_schema
from spider import spider_utils

from utils import setup_device, set_seed_everywhere

from spacy.lang.en import English

from termcolor import colored


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


def main():
    args = read_arguments_manual_inference()

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    schema_path = os.path.join(args.data_dir, "original", "tables.json")
    connection_config = {k: v for k, v in vars(args).items() if k.startswith('database')}

    schemas_raw, schemas_dict = spider_utils.load_schema(schema_path)

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

            pre_processed_with_values = _pre_process_values(pre_processed_data[0], schema_path, connection_config)

            print(f"we found the following potential values in the question: {row['values']}")

            prediction, example = _inference_semql(pre_processed_with_values, schemas_dict, model)

            print(f"Results from schema linking (question token types): {example.src_sent}")
            print(f"Results from schema linking (column types): {example.col_hot_type}")

            print(colored(f"Predicted SemQL-Tree: {prediction['model_result']}", 'magenta', attrs=['bold']))
            print()
            sql = _semql_to_sql(prediction, schemas_dict)

            print(colored(f"Transformed to SQL: {sql}", 'cyan', attrs=['bold']))
            print()
            result = _execute_query(sql, connection_config)

            print(f"Executed on the database '{args.database}'. First 10 Results: ")
            for row in result[:10]:
                print(colored(row, 'green'))

        except Exception as e:
            print("Exception: " + str(e))

        input(colored("Press [Enter] to continue with your next question.", 'red', attrs=['bold']))


if __name__ == '__main__':
    main()
