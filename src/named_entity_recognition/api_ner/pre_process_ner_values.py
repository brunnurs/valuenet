import argparse
import json
import os

from pytictoc import TicToc

from nltk import ngrams

from named_entity_recognition.api_ner.extract_values_by_heuristics import find_values_in_quota, find_ordinals, \
    find_emails
from named_entity_recognition.database_value_finder.database_value_finder import DatabaseValueFinder

DB_FOLDER = 'data/spider/original/database'
DB_SCHEMA = 'data/spider/tables.json'

all_database_value_finder = {}


def pre_process(entry):
    db_value_finder = _get_or_create_value_finder(entry)

    extracted_by_heuristic = []
    extracted_by_heuristic.extend(find_values_in_quota(entry['question']))
    extracted_by_heuristic.extend(find_ordinals(entry['question_toks']))
    extracted_by_heuristic.extend(find_emails(entry['question']))

    ner_dates = []
    ner_numbers = []
    ner_price = []
    ner_remaining = []
    for entity in entry['ner_extracted_values']['entities']:
        # for all types see https://cloud.google.com/natural-language/docs/reference/rest/v1beta2/Entity#Type
        # TODO: extend this pre-processing for e.g. ADDRESSES, PHONE_NUMBERS - see the link above.
        if entity['type'] == 'NUMBER':
            ner_numbers.append(_compose_number(entity))
        elif entity['type'] == 'DATE':
            ner_dates.extend(_compose_date(entity))
        elif entity['type'] == 'PRICE':
            ner_price.append(_compose_price(entity))
        else:
            if len(entity['name'].split(' ')) == 1:
                # just take the extracted value - without any adaptions
                ner_remaining.append(entity['name'])
            else:
                # there are multiple words in this value - create combinations out of it.
                ner_remaining.extend(_build_ngrams(entity['name']))

    ner_remaining = _find_matches_in_database(db_value_finder, ner_remaining)

    # remove duplicates
    return list(set(extracted_by_heuristic + ner_dates + ner_numbers + ner_price + ner_remaining))


def _find_matches_in_database(db_value_finder, ner_remaining):
    tic_toc = TicToc()
    tic_toc.tic()
    print(f'Find potential candiates "{ner_remaining}" in database {db_value_finder.database}')
    try:
        matching_db_values = db_value_finder.find_similar_values_in_database(ner_remaining)
        ner_remaining = list(map(lambda v: v[0], matching_db_values))
    except Exception as e:
        print(e)

    tic_toc.toc()
    return ner_remaining


def _compose_number(entity):
    # NUMBER will also detect e.g. a "one" and transform it to a 1 in the metadata
    value_as_string = entity['metadata']['value']

    if '.' in value_as_string:
        # Some floats from NER use trailing zeros - strip them away.
        return value_as_string.rstrip('0').rstrip('.')
    else:
        return value_as_string


def _compose_price(entity):
    # PRICE contains also the "currency" in the metadata. We assume that we don't need it.
    return entity['metadata']['value']


def _compose_date(entity):
    """
    This method formats returns a proper 'YYYY-MM-DD' string or a subpart of it (e.g. 'YYYY-MM') if not all information available.
    See Tests for more information.
    """
    full_date = ''
    if 'year' in entity['metadata']:
        full_date = entity['metadata']['year']

    if 'month' in entity['metadata']:
        if len(entity['metadata']['month']) == 1:
            month = '0' + entity['metadata']['month']
        else:
            month = entity['metadata']['month']

        if full_date:
            full_date = full_date + '-' + month
        else:
            full_date = month

    if 'day' in entity['metadata']:

        if len(entity['metadata']['day']) == 1:
            day = '0' + entity['metadata']['day']
        else:
            day = entity['metadata']['day']

        if full_date:
            full_date = full_date + '-' + day
        else:
            full_date = day

    # TODO: there is 4 cases where the database is a string instead of a date (e.g. "voter_2", "Voting_record" -->08/30/2015). Therefore we also deliver the value here. Fix the database.
    return [full_date, entity['name']]


def _build_ngrams(multi_token_input):
    combinations = [multi_token_input]

    # this is a rather simple splitt - might consider spaCy
    tokens = multi_token_input.split()

    for n in range(1, len(tokens)):
        # n-gram tuples can e.g. be ('John', 'Doe) and ('Doe', 'Smith')
        ngramm_tuples = ngrams(tokens, n)

        for t in ngramm_tuples:
            combinations.append(' '.join(t))

    return combinations


def are_all_values_found(expected, actual, question, query, database):
    all_values_found = True
    for value in expected:
        found = False
        for extracted_value in actual:
            if _is_value_equal(extracted_value, value):
                found = True
                break

        if not found:
            all_values_found = False
            print(
                f"Could not find '{value}' in extracted values {actual}.                                                Question: {question}                DB: {database}    Query: {query}")

    return all_values_found, expected != []


def _is_value_equal(extracted_value, expected_value):
    # there are some cases were we have a float stored in the ground truth - even though we are actually looking for an int.
    if isinstance(expected_value, float) and expected_value.is_integer():
        expected_value = int(expected_value)

    expected_value = str(expected_value)
    return expected_value == extracted_value


def _get_or_create_value_finder(entry):
    database = entry['db_id']
    if database not in all_database_value_finder:
        all_database_value_finder[database] = DatabaseValueFinder(DB_FOLDER, database, DB_SCHEMA)
    db_value_finder = all_database_value_finder[database]
    return db_value_finder


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, required=True)
    arg_parser.add_argument('--output_path', type=str, required=True)

    args = arg_parser.parse_args()

    with open(os.path.join(args.data_path), 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    entry_with_values = 0
    not_found_count = 0
    for entry in data:
        extracted_values = pre_process(entry)
        entry['ner_extracted_values_processed'] = extracted_values
        all_values_found, has_values = are_all_values_found(entry['values'], entry['ner_extracted_values_processed'],
                                                            entry['question'], entry['query'], entry['db_id'])

        if not all_values_found:
            not_found_count += 1
        if has_values:
            entry_with_values += 1
    print()
    print()
    print(
        f"Could find all values in {len(data) - not_found_count} of {len(data)} examples. {entry_with_values} entries contain values, and in {not_found_count} we could't find them")

    with open(os.path.join(args.output_path, 'ner_pre_processed_values.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f)
