import argparse
import os
import random
from pathlib import Path

import openai
from typing import List

from synthetic_data.group_pairs_to_find_templates import group_query_types, map_semql_actions_only


def ask_gpt(few_shot_samples: List[dict], sample_to_ask_for: dict, number_of_choices: int):

    # Here we assemble a prompt based on initial few-shot examples and the actual query we are interested in. A prompt might look like this:
    prompt = """Translate SQL queries to natural language questions:
    1. SQL: SELECT mailing_date FROM Documents_Mailed WHERE document_id = 7 -> Question:  What is the mail date of the document with id 7?;
    2. SQL: SELECT T1.flno FROM Flight AS T1 JOIN Aircraft AS T2 ON T1.aid  =  T2.aid WHERE T2.name  =  "Airbus A340-300" -> Question:  What are the flight numbers for the aircraft Airbus A340-300?;
    3. SQL: SELECT document_type_code FROM documents WHERE document_name  =  "David CV" -> Question: Return the type code of the document named "David CV".;
    4. SQL: SELECT T3.location_name FROM All_documents AS T1 JOIN Document_locations AS T2 ON T1.document_id  =  T2.document_id JOIN Ref_locations AS T3 ON T2.location_code  =  T3.location_code WHERE T1.document_name  =  "Robin CV" -> Question:  Show the location name for document "Robin CV".;
    5. SQL: SELECT firstname FROM teachers WHERE classroom  =  110 -> Question:  Find the first names of all the teachers that teach in classroom 110.;
    6. SQL: SELECT Zip_code FROM county WHERE County_name  =  "Howard" -> Question:"""

    few_shots = [f"{idx + 1}. SQL: {e['query']} -> Question: {e['question']}#" for idx, e in enumerate(few_shot_samples)]
    query_of_interest = f"{len(few_shots) + 1}. SQL: {sample_to_ask_for['query']} -> Question:"

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
    return response, prompt, sample_to_ask_for['question']


def main(args):
    grouped_semql, data = group_query_types(args.spider_data)

    frequent_query_types = {k:v for k,v in grouped_semql.items() if v <= 11}

    sampled_query_types = random.sample(frequent_query_types.items(),k=args.number_of_samples_to_use)

    for idx, (k, v) in enumerate(sampled_query_types):
        print(f'The following appeared {v} times: {k} Example questions are:')
        example_questions = [e for e in data if map_semql_actions_only(e['rule_label']) == k]

        # the first 10 are used for few-shot learning, the 11th is the one we ask for
        random_samples = random.sample(example_questions, k=min(11, len(example_questions)))

        response, prompt, original = ask_gpt(random_samples[:-1], random_samples[-1], args.number_of_choices)

        gpt_choices = [f"({idx}) {c['text'].strip()}" for idx, c in enumerate(response['choices'])]

        with open(Path(args.output_folder) / f'{idx}.txt', 'w') as f:
            f.write(prompt)
            f.write('\nOriginal Answer:\n')
            f.write(original)
            f.write('\nGPT-3 choices:\n')
            f.write('\n'.join(gpt_choices))




if __name__ == '__main__':
    random.seed(42)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--spider_data', type=str, default='data/spider/train.json')
    arg_parser.add_argument('--number_of_samples_to_use', type=int, default=7)
    arg_parser.add_argument('--output_folder', type=str, default='data/synthetic_data_spider')
    arg_parser.add_argument('--number_of_choices', type=int, default=8)

    args = arg_parser.parse_args()
    main(args)