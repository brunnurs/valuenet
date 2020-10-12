import argparse
import json
import os

import nltk
from pytictoc import TicToc

from named_entity_recognition.database_value_finder.database_value_finder_sqlite import DatabaseValueFinderSQLite
from named_entity_recognition.pre_process_ner_values import pre_process_ner_candidates, match_values_in_database
from preprocessing.utils import load_dataSets, wordnet_lemmatizer, symbol_filter, get_multi_token_match, \
    get_single_token_match, get_partial_match, AGG, group_symbol

import multiprocessing
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()


def lemmatize_list(names):
    # TODO: replace this splitting/lemmatization with spacy
    names_clean = []
    names_clean_list = []

    for name in names:
        x = split_lemmatize(name)
        names_clean.append(" ".join(x))
        names_clean_list.append(x)

    return names_clean, names_clean_list


def split_lemmatize(name):
    return [wordnet_lemmatizer.lemmatize(x.lower()) for x in name.split(' ')]


def add_full_column_match(token, columns_list, column_matches):
    column_ix = columns_list.index(token)
    column_matches[column_ix]['full_column_match'] = True


def add_value_match(token, columns_list, column_matches):
    column_ix = columns_list.index(token)
    column_matches[column_ix]['full_value_match'] = True


def find_values_by_database_lookup(example, ner_information, database_path, schema_path, include_primary_keys):
    example['ner_extracted_values'] = ner_information['entities']
    db_name = example['db_id']

    potential_value_candidates = pre_process_ner_candidates(example)

    database_value_finder = DatabaseValueFinderSQLite(database_path, db_name, schema_path)

    return match_values_in_database(database_value_finder, potential_value_candidates, include_primary_keys)


def schema_linking(idx, example, ner_information, schema_path, database_path):
    print()
    print(f'Process example idx: {idx}')
    print(f"Question: {example['question']}")
    print(f"SQL: {example['query']}")
    question_tokens = [wordnet_lemmatizer.lemmatize(x.lower()) for x in symbol_filter(example['question_toks'])]

    tables, table_list = lemmatize_list(example['table_names'])

    columns, columns_list = lemmatize_list(example['col_set'])

    pos_tagging = nltk.pos_tag(question_tokens)

    token_grouped = []
    token_types = []

    # this will contain what we call the "column hints" --> information how often a column has been "hit" in a question
    column_matches = [{"column_joined": '',
                       "full_column_match": False,
                       "partial_column_match": 0,
                       "full_value_match": False,
                       "partial_value_match": 0} for _ in columns]

    n_tokens = len(question_tokens)

    idx = 0
    while idx < len(question_tokens):

        # checking if we find a full column header with more than one token (e.g. "song name")
        end_idx, multi_token_column_name = get_multi_token_match(question_tokens, idx, n_tokens, columns)
        if multi_token_column_name:
            token_grouped.append(question_tokens[idx: end_idx])
            token_types.append(["col"])
            idx = end_idx

            add_full_column_match(multi_token_column_name, columns, column_matches)
            continue

        # check for table
        end_idx, table_name = get_single_token_match(question_tokens, idx, n_tokens, tables)
        if table_name:
            token_grouped.append(question_tokens[idx: end_idx])
            token_types.append(["table"])
            idx = end_idx
            continue

        # check for column
        end_idx, column_name = get_single_token_match(question_tokens, idx, n_tokens, columns)
        if column_name:
            token_grouped.append(question_tokens[idx: end_idx])
            token_types.append(["col"])
            idx = end_idx

            add_full_column_match(column_name, columns, column_matches)
            continue

        # check for partial column matches (min 2 tokens need to match)
        end_idx, column_name_extended = get_partial_match(question_tokens, idx, columns_list)
        if column_name_extended:
            token_grouped.append(
                column_name_extended)  # TODO: here we sometime add more tokens than really exists in the question. Example:
            # TODO: ['show', 'the', 'name', 'and', 'the', 'release', 'year', 'of', 'the', 'song', 'by', 'the', 'youngest', 'singer', '.'] ---> becomes --->
            # TODO: [['show'], ['the'], ['name'], ['and'], ['the'], ['song', 'release', 'year'], ['of'], ['the'], ['song'], ['by'], ['the'], ['youngest'], ['singer'], ['.']]
            # TODO: the "song release year" has been extended by song because that's the column name. Does that make sense for the transformer later?
            token_types.append(["col"])
            idx = end_idx

            add_full_column_match(column_name_extended, columns_list, column_matches)
            continue

        # check for aggregation
        end_idx, agg = get_single_token_match(question_tokens, idx, n_tokens,
                                              AGG)  # check the AGG - it's basically looking for words like "average, maximum, minimum" etc.
        if agg:
            token_grouped.append(question_tokens[idx: end_idx])
            token_types.append(["agg"])
            idx = end_idx
            continue

        if pos_tagging[idx][1] == 'RBR' or pos_tagging[idx][1] == 'JJR':  # with the help of NLTK part of speech we are
            token_grouped.append([question_tokens[idx]])  # looking for comparative words like "better", "bigger"
            token_types.append(['MORE'])
            idx += 1
            continue

        if pos_tagging[idx][1] == 'RBS' or pos_tagging[idx][1] == 'JJS':  # with the help of NLTK part of speech we are
            token_grouped.append([question_tokens[idx]])  # looking for superlative words like "best", "biggest"
            token_types.append(['MOST'])
            idx += 1
            continue

        token_grouped.append([question_tokens[idx]])
        token_types.append(['NONE'])
        idx += 1
        continue

    # This extra-loop is a bit special: we gather partial column matches by going through the question tokens and columns and finding partial matches.
    # Full matches should already have been found further up in the loop.
    # TODO not sure if that's a good thing, and even if, there might be room for improvement (e.g. a match when Table and Column has been hit).
    for column_idx, column in enumerate(columns_list):
        column_matches[column_idx]['column_joined'] = str(column)

        for question_idx, question_token in enumerate(question_tokens):
            if question_token in column:
                # if we have a match between a partial column token (e.g. "horse id") and a token in the question (e.g. "horse")
                # we will increase the counter for this column
                column_matches[column_idx]['partial_column_match'] += 1

    # Here we use the power of the base-data: if we find a potential value in the database, we mark the column we found the value in with a "full value match".
    # TODO: also use the information on table level
    include_primary_key_columns = 'id' in question_tokens
    database_matches = find_values_by_database_lookup(example, ner_information, database_path, schema_path, include_primary_key_columns)

    for value, column, table in database_matches:
        column_lemma = " ".join(split_lemmatize(column))
        column_idx = columns.index(column_lemma)
        column_matches[column_idx]['full_value_match'] = True

    return token_grouped, token_types, column_matches


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--ner_data_path', type=str, help='NER results (e.g. from Google API)', required=True)
    arg_parser.add_argument('--database_path', type=str, help='database files', required=True)
    arg_parser.add_argument('--table_path', type=str, help='schema data', required=True)
    arg_parser.add_argument('--conceptNet', type=str, help='concept net base path', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()

    # loading dataSets
    data, table = load_dataSets(args)

    with open(os.path.join(args.ner_data_path), 'r', encoding='utf-8') as json_file:
        ner_data = json.load(json_file)

    assert len(data) == len(ner_data), 'Both, NER data and actual data (e.g. ner_train.json and train.json) need to have the same amount of rows!'

    t = TicToc()
    t.tic()

    # To analyze a specific sample use this. You find the sample-idx with e.g. this code snippet:
    # [(idx, query) for idx, query in enumerate(data) if query['question'] == "THE QUESTION YOU SEARCH FOR"]
    # data = data[932:933]
    # ner_data = ner_data[932:933]

    # To better debug this code, just set "n_jobs=1"
    results = Parallel(n_jobs=NUM_CORES)(delayed(schema_linking)(idx, example, ner_information, args.table_path, args.database_path) for idx, (example, ner_information) in enumerate(zip(data, ner_data)))
    all_token_grouped, all_token_types, all_column_matches = zip(*results)

    for example, token_grouped, token_types, column_matches in zip(data, all_token_grouped, all_token_types, all_column_matches):

        example['question_arg'] = token_grouped
        example['question_arg_type'] = token_types
        example['column_matches'] = column_matches

    t.toc(msg="Total pre-processing took")

    with open(args.output, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
