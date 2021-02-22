from pytictoc import TicToc

from nltk import ngrams

from named_entity_recognition.database_value_finder.database_value_finder_sqlite import DatabaseValueFinderSQLite
from named_entity_recognition.handcrafted_heuristics import find_values_in_quote, find_ordinals, \
    find_emails, find_genders, find_null_empty_values, find_variety_of_common_mentionings, find_special_codes, \
    find_single_letters, find_capitalized_words, find_months, find_location_abbreviations

from named_entity_recognition.ner_extraction_data_dto import NerExtractionData

all_database_value_finder = {}


def _get_or_create_value_finder(database, database_folder, db_schema):
    if database not in all_database_value_finder:
        all_database_value_finder[database] = DatabaseValueFinderSQLite(database_folder, database, db_schema)
    db_value_finder = all_database_value_finder[database]
    return db_value_finder


def pre_process_ner_candidates(ner_extracted_values, question, question_tokens):

    extracted_data = NerExtractionData([], [], [], [], [], [], [], [], [], [], [], [], [], [], [])

    extracted_data.heuristic_values_in_quote.extend(find_values_in_quote(question))
    extracted_data.heuristic_ordinals.extend(find_ordinals(question_tokens))
    extracted_data.heuristics_emails.extend(find_emails(question))
    extracted_data.heuristics_genders.extend(find_genders(question_tokens))
    extracted_data.heuristics_null_empty.extend(find_null_empty_values(question_tokens))
    extracted_data.heuristics_variety_common_mentionings.extend(find_variety_of_common_mentionings(question_tokens))
    extracted_data.heuristics_special_codes.extend(find_special_codes(question))
    extracted_data.heuristics_single_letters.extend(find_single_letters(question))
    extracted_data.heuristics_capitalized_words.extend(find_capitalized_words(question))
    extracted_data.heuristics_months.extend(find_months(question_tokens))
    extracted_data.heuristics_location_abbreviations.extend(find_location_abbreviations(question_tokens, question))

    for entity in ner_extracted_values:
        # for all types see https://cloud.google.com/natural-language/docs/reference/rest/v1beta2/Entity#Type
        # TODO: extend this pre-processing for e.g. ADDRESSES, PHONE_NUMBERS - see the link above.
        if entity['type'] == 'NUMBER':
            extracted_data.ner_numbers.append(_compose_number(entity))
        elif entity['type'] == 'DATE':
            extracted_data.ner_dates.extend(_compose_date(entity))
        elif entity['type'] == 'PRICE':
            extracted_data.ner_prices.append(_compose_price(entity))
        else:
            if len(entity['name'].split(' ')) == 1:
                # just take the extracted value - without any adaptions
                extracted_data.ner_remaining.append(entity['name'])
            else:
                # there are multiple words in this value - create combinations out of it.
                extracted_data.ner_remaining.extend(_build_ngrams(entity['name']))

    return extracted_data


def match_values_in_database(db_value_finder, extracted_data, include_primary_keys):

    # NOTE: adapting this thresholds should always be done empirically and is heavily depending on the chosen similarity metric.
    # have a look at the script find_optimal_similarity_threshold.py to find optimal thresholds.
    exact_match = db_value_finder.exact_match_threshold
    high_similarity = db_value_finder.high_similarity_threshold
    medium_similarity = db_value_finder.medium_similarity_threshold

    # depending on the candidate type we set a different tolerance value for similarity matching with db-values.
    # Remember: 1.0 is looking for exact matches only. Also remember: we do lower-case only comparison, so 'Male' and 'male' will match with 1.0
    candidates = []
    # With values in quote we are a bit tolerant. Important: we keep this values anyway, as the are often used in fuzzy LIKE searches.
    _add_without_duplicates([(quote, high_similarity) for quote in extracted_data.heuristic_values_in_quote], candidates)
    # Gender values we only want exact matches.
    _add_without_duplicates([(gender, exact_match) for gender in extracted_data.heuristics_genders], candidates)
    _add_without_duplicates([(common_mentionings, high_similarity) for common_mentionings in extracted_data.heuristics_variety_common_mentionings], candidates)
    # a special code should match exactly
    _add_without_duplicates([(special_code, exact_match) for special_code in extracted_data.heuristics_special_codes], candidates)
    _add_without_duplicates([(capitalized_word, medium_similarity) for capitalized_word in extracted_data.heuristics_capitalized_words], candidates)
    _add_without_duplicates([(location, high_similarity) for location in extracted_data.heuristics_location_abbreviations], candidates)

    # important: in addition to all the handcrafted features, also take all values from the NER which aren't known dates/numbers/prices
    _add_without_duplicates([(ner_value, medium_similarity) for ner_value in extracted_data.ner_remaining], candidates)

    _add_without_duplicates([(ordinal, exact_match) for ordinal in extracted_data.heuristic_ordinals], candidates)
    _add_without_duplicates([(email, high_similarity) for email in extracted_data.heuristics_emails], candidates)
    _add_without_duplicates([(single_letter, exact_match) for single_letter in extracted_data.heuristics_single_letters], candidates)
    _add_without_duplicates([(ner_date, exact_match) for ner_date in extracted_data.ner_dates], candidates)
    _add_without_duplicates([(ner_number, exact_match) for ner_number in extracted_data.ner_numbers], candidates)
    _add_without_duplicates([(ner_price, exact_match) for ner_price in extracted_data.ner_prices], candidates)

    tic_toc = TicToc()
    tic_toc.tic()
    print(f'Look for potential candidates "{candidates}" in database {db_value_finder.database} (include primary keys: {include_primary_keys})')
    matching_db_values = db_value_finder.find_similar_values_in_database(candidates, include_primary_keys)
    print(f'Confirmed the following candidates "{matching_db_values}"')
    tic_toc.toc()

    return matching_db_values


def _add_without_duplicates(new_candidates, candidates):
    for value, tolerance in new_candidates:
        existing_candidate = next(filter(lambda value_tolerance: value_tolerance[0] == value, candidates), None)
        if existing_candidate:
            existing_value, existing_tolerance = existing_candidate
            if existing_tolerance < tolerance:
                candidates.remove((existing_value, existing_tolerance))
                candidates.append((value, tolerance))
        else:
            candidates.append((value, tolerance))


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


def add_non_found_values(expected_values, candidates_original):
    candidates = candidates_original.copy()

    all_found = True
    for value in expected_values:
        found = False
        for extracted_value in candidates:
            if _is_value_equal(extracted_value, value):
                found = True
                break

        if not found:
            all_found = False
            print(f"Could not find '{value}' in extracted values '{candidates}'. We add it from the ground truth.")
            candidates.append(value)

    return candidates, all_found, len(expected_values)


def _is_value_equal(extracted_value, expected_value):
    # there are some cases were we have a float stored in the ground truth - even though we are actually looking for an int.
    if isinstance(expected_value, float) and expected_value.is_integer():
        expected_value = int(expected_value)

    expected_value = str(expected_value)

    return expected_value == extracted_value


def compare_questions(row, ner_information):
    if len(row['question']) != len(ner_information['question']):
        return row['question'][:-2] != ner_information['question'][:-1]
    else:
        return row['question'] != ner_information['question']


# TODO: this part of the code is not used currently, as the pre-processing of ner-values is done as part of the standard pre-processing. See pre_process.py.
# if __name__ == '__main__':
#     arg_parser = argparse.ArgumentParser()
#     arg_parser.add_argument('--data_path', type=str, required=True)
#     arg_parser.add_argument('--ner_data_path', type=str, required=True)
#     arg_parser.add_argument('--output_path', type=str, required=True)
#     arg_parser.add_argument('--database_folder', type=str, default='data/spider/original/database')
#     arg_parser.add_argument('--database_schema', type=str, default='data/spider/original/tables.json')
#
#     args = arg_parser.parse_args()
#
#     with open(os.path.join(args.data_path), 'r', encoding='utf-8') as json_file:
#         data = json.load(json_file)
#
#     with open(os.path.join(args.ner_data_path), 'r', encoding='utf-8') as json_file:
#         ner_data = json.load(json_file)
#
#     assert len(data) == len(ner_data), 'Both, NER data and actual data (e.g. ner_train.json and preprocessed_train.json) need to have the same amount of rows!'
#
#     # add both, the ner-extracted values and the actual values (extracted from the SQL-ground truth) to the data file.
#     for idx, (row, ner_information) in enumerate(zip(data, ner_data)):
#         if compare_questions(row, ner_information):
#             print(f'{idx}       {row["question"]}               {ner_information["question"]}')
#         row['ner_extracted_values'] = ner_information['entities']
#         row['values'] = ner_information['values']
#
#     entry_with_values = 0
#     not_found_count = 0
#     total_expected_value_count = 0
#
#     # here we pre-process the NER results and add further values by handcrafted heuristics
#     extracted_values = [pre_process_ner_candidates(row) for idx, row in enumerate(data)]
#     print("Preprocessed all NER values and applied handcrafted handcrafted heuristics. "
#           "Next Step: matching values in database. This might take a while.")
#     print()
#
#     # here we takes the pre-processed values and try to match them in the database. As this process is very time-consuming,
#     # we parallelize it. Important: Parallel() maintains the order of the input data!
#     n_cores = multiprocessing.cpu_count()
#     values_matched_with_database = Parallel(n_jobs=n_cores)(
#         delayed(match_values_in_database)(row['db_id'], extracted_value, args.database_folder, args.database_schema) for extracted_value, row
#         in zip(extracted_values, data))
#     print("Scanned all databases for matching values.")
#     print()
#
#     for row, value_candidates in zip(data, values_matched_with_database):
#         # this method is basically cheating: if we don't find a value in the values candidates, we add it from the ground truth.
#         # This makes sense for training, as we don't want to reduce the training samples because of non-found values. We also mark
#         # the samples where not all values could get extracted, so we can manually fail them during evaluation.
#         value_candidates_adjusted, all_values_found, n_expected_values = add_non_found_values(row['values'],
#                                                                                               value_candidates,
#                                                                                               row['question'],
#                                                                                               row['query'],
#                                                                                               row['db_id'])
#
#         row['ner_extracted_values_processed'] = value_candidates_adjusted
#         row['all_values_found'] = all_values_found
#
#         if not all_values_found:
#             not_found_count += 1
#
#         if n_expected_values > 0:
#             entry_with_values += 1
#
#         total_expected_value_count += n_expected_values
#
#     print()
#     print()
#     print(
#         f"Could find all values in {len(data) - not_found_count} of {len(data)} examples. {entry_with_values} entries "
#         f"contain values, and in {not_found_count} we could't find them. There is a total of {total_expected_value_count} values in this dataset")
#
#     with open(os.path.join(args.output_path), 'w', encoding='utf-8') as f:
#         json.dump(data, f, indent=2)
