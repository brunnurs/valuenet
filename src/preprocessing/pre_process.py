import argparse
import json
import os
import pickle

import nltk

from preprocessing.utils import load_dataSets, wordnet_lemmatizer, symbol_filter, get_multi_token_match, \
    get_single_token_match, get_partial_match, AGG, group_symbol, group_values, group_digital, num2year


def lemmatize_list(names):
    # TODO: replace this splitting/lemmatization with spacy
    names_clean = []
    names_clean_list = []

    for name in names:
        x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in name.split(' ')]
        names_clean.append(" ".join(x))
        names_clean_list.append(x)

    return names_clean, names_clean_list


def schema_linking(example, related_to_concept_net, is_a_concept_net):
    question_tokens = [wordnet_lemmatizer.lemmatize(x.lower()) for x in symbol_filter(example['question_toks'])]

    tables, table_list = lemmatize_list(example['table_names'])

    columns, columns_list = lemmatize_list(example['col_set'])

    pos_tagging = nltk.pos_tag(question_tokens)

    token_grouped = []
    token_types = []

    # this will contain what we call the "column hints" --> information how often a column has been "hit" in a question
    column_matches = [{"full_column_match": False,
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

        # string match for Time Format
        if num2year(question_tokens[idx]):
            end_idx, header = get_single_token_match(question_tokens, idx, n_tokens, columns)
            if header:
                token_grouped.append(question_tokens[idx: end_idx])
                token_types.append(["col"])
                idx = end_idx
                continue

        def get_concept_result(toks, graph):
            for begin_id in range(0, len(toks)):
                for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
                    tmp_query = "_".join(toks[begin_id:r_ind])
                    if tmp_query in graph:
                        mi = graph[tmp_query]
                        for col in example['col_set']:
                            if col in mi:
                                return col

        end_idx, symbol = group_symbol(question_tokens, idx, n_tokens)
        if symbol:
            tmp_toks = [x for x in question_tokens[idx: end_idx]]
            assert len(tmp_toks) > 0, print(symbol, question_tokens)
            pro_result = get_concept_result(tmp_toks, is_a_concept_net)
            if pro_result is None:
                pro_result = get_concept_result(tmp_toks, related_to_concept_net)
            if pro_result is None:
                pro_result = "NONE"
            for tmp in tmp_toks:
                token_grouped.append([tmp])
                token_types.append([pro_result])
                pro_result = "NONE"
            idx = end_idx
            continue

        end_idx, values = group_values(question_tokens, idx, n_tokens)
        if values and (len(values) > 1 or question_tokens[idx - 1] not in ['?', '.']):
            tmp_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_tokens[idx: end_idx] if x.isalnum() is True]
            assert len(tmp_toks) > 0, print(question_tokens[idx: end_idx], values, question_tokens, idx, end_idx)
            pro_result = get_concept_result(tmp_toks, is_a_concept_net)
            if pro_result is None:
                pro_result = get_concept_result(tmp_toks, related_to_concept_net)
            if pro_result is None:
                pro_result = "NONE"
            for tmp in tmp_toks:
                token_grouped.append([tmp])
                token_types.append([pro_result])
                pro_result = "NONE"
            idx = end_idx
            continue

        result = group_digital(question_tokens, idx)
        if result is True:
            token_grouped.append(question_tokens[idx: idx + 1])
            token_types.append(["value"])
            idx += 1
            continue
        if question_tokens[idx] == ['ha']:
            question_tokens[idx] = ['have']

        token_grouped.append([question_tokens[idx]])
        token_types.append(['NONE'])
        idx += 1
        continue

    # TODO this code should get removed as soon as we know that everything still work the same. We need to integrate this in the loop above
    # TODO and figure out if it is even correct to to the next few lines. In my opinion we should not just put weight on columns where we for example know
    # TODO exactly that they are tables.
    for column_idx, column in enumerate(columns_list):
        for question_idx, question_token in enumerate(question_tokens):
            if question_token in column:
                # if we have a match between a partial column token (e.g. "horse id") and a token in the question (e.g. "horse")
                # we will increase the counter for this column
                column_matches[column_idx]['partial_column_match'] += 1

    for token, token_type in zip(token_grouped, token_types):
        if token_type == ["col"]:
            column_ix = columns_list.index(token)
            column_matches[column_ix]['full_column_match'] = True
        elif token_type == ['NONE']:
            pass
        elif token_type == ['table']:
            pass
        elif token_type == ['agg']:
            pass
        elif token_type == ['MORE']:
            pass
        elif token_type == ['MOST']:
            pass
        elif token_type == ['value']:
            pass
        else:
            if len(token_type) == 1:
                value_type_which_should_be_a_column_name = token_type[0]
                # if the token type (col_probase) is e.g. "time of day" (because the word was "night" and we found with ConceptNet it is a "time of day")
                # we try to find a column with this name. If we find it --> we have an "Exact match", so we mark the matching column with a 5 (in the "col_set_type" array)
                # NOTE: we could do this much smarter by using the table values.
                column_ix = columns.index(value_type_which_should_be_a_column_name)
                column_matches[column_ix]['full_value_match'] = True
            else:
                raise ValueError("Investigate this!")

    return token_grouped, token_types, column_matches


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--conceptNet', type=str, help='concept net base path', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()

    # loading dataSets
    data, table = load_dataSets(args)

    with open(os.path.join(args.conceptNet, 'english_RelatedTo.pkl'), 'rb') as f:
        related_to_concept_net = pickle.load(f)

    with open(os.path.join(args.conceptNet, 'english_IsA.pkl'), 'rb') as f:
        is_a_concept_net = pickle.load(f)

    for example in data:
        token_grouped, token_types, column_matches = schema_linking(example, related_to_concept_net, is_a_concept_net)

        example['question_arg'] = token_grouped
        example['question_arg_type'] = token_types
        example['column_matches'] = column_matches

    with open(args.output, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
