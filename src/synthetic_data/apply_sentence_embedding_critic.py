import argparse
import json
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, util
from collections import Counter


def read_generative_choices(path):
    with open(path) as f:
        lines = f.readlines()

    try:
        original_query = lines[lines.index("Original Query:\n") + 1]
    except:
        original_query = '---'

    generative_choices = []
    for i in range(0, 8):
        choice_line = filter(lambda l: l.startswith(f'({i})'), lines)
        choice = next(choice_line)[4:]
        choice = choice.strip()

        generative_choices.append(choice)

    return generative_choices, original_query.strip(), lines


def rank_by_aggregated_pairwise_similarity(choices: List[str], model: SentenceTransformer):
    paraphrases = util.paraphrase_mining(model, choices)

    choice_scores = {s: 0 for s in choices}
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        choice_scores[choices[i]] += score
        choice_scores[choices[j]] += score

    c = Counter(choice_scores)
    return c.most_common()


def print_reranked(file_path: Path, original_lines: List[str], choices_reranked: List[Tuple[str, float]]):
    re_ranked = [f'{v:.3f}  {c}\n' for c, v in choices_reranked]

    new_file = Path(file_path.parent / f'_{file_path.name}')

    new_file_content = f"""{''.join(original_lines)}


Re-ranked choices:
{''.join(re_ranked)}
"""

    new_file.write_text(new_file_content)


def main(args):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    samples = []

    for idx, path in enumerate(Path(args.input_data_path).glob('*.txt')):
        choices, original_sql_query, original_file_content = read_generative_choices(path)

        choice_reranked = rank_by_aggregated_pairwise_similarity(choices, model)

        print_reranked(path, original_file_content, choice_reranked)

        print(f'{idx}: {original_sql_query}')
        print(choices)
        print(choice_reranked)
        print()
        print()

        # we wanna keep both, the first and the second choice after re-ranking
        samples.append({
            'db_id': args.db_id,
            'id': f'{idx}_1',
            'user': "gpt-3",
            'question': choice_reranked[0][0],
            'query': original_sql_query
        })

        samples.append({
            'db_id': args.db_id,
            'id': f'{idx}_2',
            'user': "gpt-3",
            'question': choice_reranked[1][0],
            'query': original_sql_query
        })

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_data_path', type=str, default='data/cordis/generative/generated')
    arg_parser.add_argument('--output_file', type=str, default='data/cordis/generative/all_synthetic.json')
    arg_parser.add_argument('--db_id', type=str, default='cordis_temporary')

    args = arg_parser.parse_args()
    main(args)