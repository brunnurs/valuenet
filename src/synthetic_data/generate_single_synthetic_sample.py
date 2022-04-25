import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

import openai
from typing import List

from synthetic_data.group_pairs_to_find_templates import group_query_types, map_semql_actions_only
from synthetic_data.sample_queries.sample_query import sample_query
from tools.transform_generative_schema import GenerativeSchema


def ask_gpt(sample: str, number_of_choices: int, model_id: str):

    prompt = sample + '\n\n###\n\n'
    response = openai.Completion.create(
        model=model_id,
        prompt=prompt,
        # top_p=0.9,
        max_tokens=128,
        n=number_of_choices,
        # frequency_penalty=0.5,
        # presence_penalty=0.5,
        stop=["\n"]
    )

    print(response)
    return response, prompt


def single_request(args):

    prompt = """
SELECT funding_schemes.title FROM funding_schemes JOIN projects ON funding_schemes.code = projects.ec_fund_scheme WHERE projects.unics_id = 156767
""".strip()

    response, prompt = ask_gpt(prompt,
                               number_of_choices=args.number_of_choices,
                               model_id=args.gpt3_finetuned_model)

    gpt_choices = [f"({idx}) {c['text'].strip()}" for idx, c in enumerate(response['choices'])]

    with open(Path(args.output_folder) / f'999.txt', 'w') as f:
        f.write(prompt)
        f.write('\nGPT-3 choices:\n')
        f.write('\n'.join(gpt_choices))


if __name__ == '__main__':
    random.seed(42)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, default='data/cordis')
    arg_parser.add_argument('--output_folder', type=str, default='data/cordis/generative')
    arg_parser.add_argument('--number_of_choices', type=int, default=8)
    arg_parser.add_argument('--gpt3_finetuned_model', type=str, default='davinci:ft-personal-2022-01-17-10-28-10')

    args = arg_parser.parse_args()
    single_request(args)