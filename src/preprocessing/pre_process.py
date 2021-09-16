import argparse
import json
import os

import nltk
from pytictoc import TicToc

from named_entity_recognition.database_value_finder.database_value_finder_postgresql import \
    DatabaseValueFinderPostgreSQL
from named_entity_recognition.database_value_finder.database_value_finder_sqlite import DatabaseValueFinderSQLite
from named_entity_recognition.pre_process_ner_values import pre_process_ner_candidates, match_values_in_database, \
    add_non_found_values
from preprocessing.utils import load_dataSets, wordnet_lemmatizer, symbol_filter, get_multi_token_match, \
    get_single_token_match, get_partial_match, AGG

import multiprocessing
from joblib import Parallel, delayed

# keep always a few cores free
NUM_CORES = max(multiprocessing.cpu_count() - 2, 1)


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


def build_db_value_finder(db_name, schema_path, args):
    # If a database path is provided, we assume an SQLite database and initialize the DatabaseValueFinderSQLite.
    # In case there is no database path we assume a bunch of connection details for the DatabaseValueFinderPostgreSQL-

    if 'database_path' in args and args.database_path:
        return DatabaseValueFinderSQLite(args.database_path, db_name, schema_path)
    else:
        connection_config = {k: v for k, v in vars(args).items() if k.startswith('database')}
        connection_config['database'] = db_name

        return DatabaseValueFinderPostgreSQL(db_name, schema_path, connection_config)


def add_likely_value_candidates(value_candidates, potential_value_candidates):
    """
    Some values can by definition not get found in the database - take the example "how many dogs are older than 20 years?"
    The value 20 will most probably not get found in the database, but maybe 21 or 32. Similar cases appear when a user asks for something
    that does not exist (e.g. "give me all bills for Christian Benz" - and there is no "Christian Benz")
    We therefore add certain values always to the value_candidates
    """
    return list(set(
        potential_value_candidates.heuristic_values_in_quote +  # we put in values in quote a second time as those values are often fuzzy strings.
        potential_value_candidates.heuristic_ordinals +
        potential_value_candidates.heuristics_emails +
        potential_value_candidates.heuristics_null_empty +
        potential_value_candidates.heuristics_single_letters +
        potential_value_candidates.heuristics_months +
        potential_value_candidates.ner_dates +
        potential_value_candidates.ner_numbers +
        potential_value_candidates.ner_prices +
        potential_value_candidates.heuristics_special_codes +
        potential_value_candidates.heuristics_capitalized_words +
        value_candidates))


def lookup_database(example, ner_information, columns, question_tokens, column_matches, database_value_finder,
                    add_values_from_ground_truth):
    """
    Now we use the base data (database) for two things:
    * to put together a list of values from which the neural network later hase to pick the right one.
    * to get "hints" which column should get selected, based on if we find a value in this column.
    As as input we use the entities extracted by the NER and then boil it down with the help of the base data (database).
    """

    potential_value_candidates = pre_process_ner_candidates(ner_information['entities'], example['question'],
                                                            example['question_toks'])

    # Here we use the power of the base-data: if we find a potential value in the database, we mark the column we found the value in with a "full value match".
    # TODO: also use the information on table level
    include_primary_key_columns = 'id' in question_tokens

    # here we do the actual database lookup
    database_matches = match_values_in_database(database_value_finder, potential_value_candidates,
                                                include_primary_key_columns)

    # and add the hint to the corresponding column
    for value, column, table in database_matches:
        column_lemma = " ".join(split_lemmatize(column))
        column_idx = columns.index(column_lemma)
        column_matches[column_idx]['full_value_match'] = True

    #  Now we basically just have to return all the potential values - the neural network later has to pick the correct one.
    value_candidates = list(map(lambda v: v[0], database_matches))

    # Some values can by definition not get found in the database - take the example "how many dogs are older than 20 years?"
    # The value 20 will most probably not get found in the database, but maybe 21 or 32. Similar cases appear when a user asks for something
    # that does not exist (e.g. "give me all bills for Christian Benz" - and there is no "Christian Benz")
    # We therefore add certain values always to the value_candidates
    value_candidates = add_likely_value_candidates(value_candidates, potential_value_candidates)

    if add_values_from_ground_truth:
        # but what if we can't find all values, because e.g. the NER does not return the correct candidates?:
        # if we don't find a value in the values candidates, we add it from the ground truth. Be aware that is is basically cheating.
        # This makes sense though for training, as we don't want to reduce the training samples because of non-found values. We also mark
        # the samples where not all values could get extracted, so we can manually fail them during evaluation.
        value_candidates, all_values_found, _ = add_non_found_values(ner_information['values'], value_candidates)

        if not all_values_found:
            print(str(potential_value_candidates))
            print(ner_information['values'])
    else:
        all_values_found = True

    return value_candidates, all_values_found, column_matches


def pre_process(idx, example, ner_information, db_value_finder, is_training):
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

    # a lot of interesting stuff happens here - make sure you are aware of it!
    value_candidates, all_values_found, column_matches = lookup_database(example, ner_information, columns,
                                                                         question_tokens, column_matches,
                                                                         db_value_finder,
                                                                         add_values_from_ground_truth=is_training)

    return token_grouped, token_types, column_matches, value_candidates, all_values_found


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--ner_data_path', type=str, help='NER results (e.g. from Google API), including actual values extracted from SQL', required=True)
    arg_parser.add_argument('--database_path', type=str, help='Database file in case of SQLite', required=False)
    arg_parser.add_argument('--database_host', type=str, help='Database host in case of PostgreSQL', required=False)
    arg_parser.add_argument('--database_port', type=str, help='Host port in case of PostgreSQL', required=False)
    arg_parser.add_argument('--database_user', type=str, help='Database user in case of PostgreSQL', required=False)
    arg_parser.add_argument('--database_password', type=str, help='Database password in case of PostgreSQL', required=False)
    arg_parser.add_argument('--database_schema', type=str, help='Database schema in case of PostgreSQL', required=False)
    arg_parser.add_argument('--table_path', type=str, help='schema data', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()

    # loading dataSets
    data, table = load_dataSets(args)

    with open(os.path.join(args.ner_data_path), 'r', encoding='utf-8') as json_file:
        ner_data = json.load(json_file)

    assert len(data) == len(
        ner_data), 'Both, NER data and actual data (e.g. ner_train.json and train.json) need to have the same amount of rows!'

    not_found_count = 0

    t = TicToc()
    t.tic()

    # To analyze a specific sample use this. You find the sample-idx with e.g. this code snippet:
    # [(idx, query) for idx, query in enumerate(data) if query['question'] == "THE QUESTION YOU SEARCH FOR"]
    # data = data[7646:7647]
    # ner_data = ner_data[7646:7647]

    results = Parallel(n_jobs=NUM_CORES)(delayed(pre_process)(idx, example, ner_information,
                                                              build_db_value_finder(example['db_id'], args.table_path, args),
                                                              is_training=True) for idx, (example, ner_information) in
                                         enumerate(zip(data, ner_data)))
    # To better debug this code, use the non-parallelized version of the code
    # results = [pre_process(idx, example, ner_information, build_db_value_finder(args.database_path, example['db_id'], args.table_path), is_training=True) for idx, (example, ner_information) in enumerate(zip(data, ner_data))]

    all_token_grouped, all_token_types, all_column_matches, all_value_candidates, all_complete_values_found = zip(
        *results)

    for example, token_grouped, token_types, column_matches, value_candidates, complete_values_found in zip(data,
                                                                                                            all_token_grouped,
                                                                                                            all_token_types,
                                                                                                            all_column_matches,
                                                                                                            all_value_candidates,
                                                                                                            all_complete_values_found):

        # this are the only additional information we store after pre-processing
        example['question_arg'] = token_grouped
        example['question_arg_type'] = token_types
        example['column_matches'] = column_matches
        example['ner_extracted_values_processed'] = value_candidates
        example['all_values_found'] = complete_values_found

        if not complete_values_found:
            not_found_count += 1

    t.toc(msg="Total pre-processing took")
    print(
        f"Could not find all values in {not_found_count} examples. All examples where values could not get extracted, will get disable on evaluation")

    with open(args.output, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
