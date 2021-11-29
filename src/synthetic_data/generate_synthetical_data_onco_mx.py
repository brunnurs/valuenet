import argparse
import csv
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

import openai
from typing import List

from synthetic_data.group_pairs_to_find_templates import group_query_types, map_semql_actions_only
from synthetic_data.sample_queries.sample_query import sample_query


def ask_gpt(few_shot_samples: List[dict], sample_to_ask_for: str, number_of_choices: int):

    # Here we assemble a prompt based on initial few-shot examples and the actual query we are interested in. A prompt might look like this:
    prompt = """Translate SQL queries to natural language questions:
    1. SQL: SELECT mailing_date FROM Documents_Mailed WHERE document_id = 7 -> Question:  What is the mail date of the document with id 7?;
    2. SQL: SELECT T1.flno FROM Flight AS T1 JOIN Aircraft AS T2 ON T1.aid  =  T2.aid WHERE T2.name  =  "Airbus A340-300" -> Question:  What are the flight numbers for the aircraft Airbus A340-300?;
    3. SQL: SELECT document_type_code FROM documents WHERE document_name  =  "David CV" -> Question: Return the type code of the document named "David CV".;
    4. SQL: SELECT T3.location_name FROM All_documents AS T1 JOIN Document_locations AS T2 ON T1.document_id  =  T2.document_id JOIN Ref_locations AS T3 ON T2.location_code  =  T3.location_code WHERE T1.document_name  =  "Robin CV" -> Question:  Show the location name for document "Robin CV".;
    5. SQL: SELECT firstname FROM teachers WHERE classroom  =  110 -> Question:  Find the first names of all the teachers that teach in classroom 110.;
    6. SQL: SELECT Zip_code FROM county WHERE County_name  =  "Howard" -> Question:"""

    few_shots = [f"{idx + 1}. SQL: {e['query']} -> Question: {e['question']}#" for idx, e in enumerate(few_shot_samples)]
    query_of_interest = f"{len(few_shots) + 1}. SQL: {sample_to_ask_for} -> Question:"

    prompt = "Translate SQL queries to natural language questions:\n" + '\n'.join(few_shots) + '\n' + query_of_interest

    print(prompt)
    print()
    print()
    print()

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        top_p=0.9,
        max_tokens=128,
        n=number_of_choices,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stop=["#"]
    )

    print(response)
    return response, prompt


def load_handmade_samples(data_path: Path):
    # load hand-made, OncoMX specific training data. We use them to prompt-engineer GPT-3
    samples = csv.DictReader(open(data_path / 'generative' / 'sample_queries.csv'))

    # replace new lines with a space to be more consistent
    example_queries = [{'query': re.sub(r'(\r\n|\r|\n)', ' ', row['query']), 'question': row['question']}
                       for row in samples]
    return example_queries


def main(args):
    # group query types based on spider data
    grouped_semql, data = group_query_types(args.spider_data)

    example_queries = load_handmade_samples(Path(args.data_path))

    db_config = SimpleNamespace(database=args.database,
                                db_user=args.db_user,
                                db_password=args.db_password,
                                db_host=args.db_host,
                                db_port=args.db_port,
                                db_options=args.db_options)

    # only consider the 3 most common queries for now
    for idx, (query_type, _) in enumerate(grouped_semql.most_common(3)):
        sampled_query = sample_query(query_type, data, Path(args.data_path), db_config)

        print(f'We sample query "{sampled_query}" for query type "{query_type}".')

        # We use a few handmade OncoMX samples to prompt-engineer the few-shot learning
        random_samples = random.sample(example_queries, k=args.number_of_samples_to_use)

        response, prompt = ask_gpt(random_samples, sampled_query, args.number_of_choices)

        gpt_choices = [f"({idx}) {c['text'].strip()}" for idx, c in enumerate(response['choices'])]

        with open(Path(args.output_folder) / f'{idx}.txt', 'w') as f:
            f.write(prompt)
            f.write('\nOriginal Answer:\n')
            f.write(sampled_query)
            f.write('\nGPT-3 choices:\n')
            f.write('\n'.join(gpt_choices))


if __name__ == '__main__':
    random.seed(42)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--spider_data', type=str, default='data/spider/train.json')
    arg_parser.add_argument('--data_path', type=str, default='data/oncomx')
    arg_parser.add_argument('--number_of_samples_to_use', type=int, default=10)
    arg_parser.add_argument('--output_folder', type=str, default='data/synthetic_data_oncomx')
    arg_parser.add_argument('--number_of_choices', type=int, default=8)

    arg_parser.add_argument('--database', type=str, default='oncomx_v1_0_25_small')
    arg_parser.add_argument('--db_user', type=str, default='postgres')
    arg_parser.add_argument('--db_password', type=str, default='vdS83DJSQz2xQ')
    arg_parser.add_argument('--db_host', type=str, default='testbed.inode.igd.fraunhofer.de')
    arg_parser.add_argument('--db_port', type=str, default='18001')
    arg_parser.add_argument('--db_options', type=str, default=f"-c search_path=oncomx_v1_0_25,public")

    args = arg_parser.parse_args()
    main(args)