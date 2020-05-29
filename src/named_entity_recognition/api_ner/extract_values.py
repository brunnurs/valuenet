import argparse
import json
import os

from named_entity_recognition.api_ner.google_api_repository import remote_named_entity_recognition

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, required=True)
    arg_parser.add_argument('--output_path', type=str, required=True)

    args = arg_parser.parse_args()

    with open(os.path.join(args.data_path), 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    error_count = 0
    ner_data = []
    for doc in data:
        extracted_values = remote_named_entity_recognition(doc['question'])
        if extracted_values:
            ner_data.append({
                'entities': extracted_values['entities'],
                'language': extracted_values['language'],
                'question': doc['question']
            })
        else:
            error_count += 1

    with open(os.path.join(args.output_path), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print("Extracted {} values. {} requests failed.".format(len(data), error_count))
