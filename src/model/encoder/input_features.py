import torch
from more_itertools import flatten

SEGMENT_ID_QUESTION = 0
SEGMENT_ID_SCHEMA = 1


def encode_input(question_spans, column_names, table_names, tokenizer, max_length_model, device):
    all_input_ids = []
    all_attention_mask = []
    all_segment_ids = []

    all_question_span_lengths = []
    all_column_token_lengths = []
    all_table_token_lengths = []

    for question, columns, tables in zip(question_spans, column_names, table_names):
        question_tokens, question_span_lengths, question_segment_ids = _tokenize_question(question, tokenizer)
        all_question_span_lengths.append(question_span_lengths)

        columns_tokens, column_token_lengths, columns_segment_ids = _tokenize_column_names(columns, tokenizer)
        # TODO: this is an exception case (db-id: "baseball_1") which leads to too many tokens. Therefore we don't sub-tokenize it
        if len(columns_tokens) == 433:
            columns_tokens, column_token_lengths, columns_segment_ids = _tokenize_column_names(columns, tokenizer,  do_sub_tokenizing=False)
            # print("Found a 'baseball_1' case!")

        all_column_token_lengths.append(column_token_lengths)

        table_tokens, table_token_lengths, table_segment_ids = _tokenize_table_names(tables, tokenizer)
        all_table_token_lengths.append(table_token_lengths)

        assert sum(question_span_lengths) + sum(column_token_lengths) + sum(table_token_lengths) == \
               len(question_tokens) + len(columns_tokens) + len(table_tokens)

        tokens = question_tokens + columns_tokens + table_tokens
        if len(tokens) > max_length_model:
            print("################### ATTENTION! Example too long ({}). Question-len: {}, column-len:{}, table-len: {} ".format(len(tokens), len(question_tokens), len(columns_tokens), len(table_tokens)))
            print(question)
            print(columns)
            print(tables)

        segment_ids = question_segment_ids + columns_segment_ids + table_segment_ids
        # not sure here if "tokenizer.mask_token_id" or just a simple 1...
        attention_mask = [1] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        all_input_ids.append(input_ids)
        all_segment_ids.append(segment_ids)
        all_attention_mask.append(attention_mask)

    max_length_data = max(map(lambda ids: len(ids), all_input_ids))

    for input_ids, segment_ids, attention_mask in zip(all_input_ids, all_segment_ids, all_attention_mask):
        _padd_input(input_ids, segment_ids, attention_mask, max_length_data, tokenizer)

    input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    segment_ids_tensor = torch.tensor(all_segment_ids, dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor(all_attention_mask, dtype=torch.long).to(device)

    return input_ids_tensor, attention_mask_tensor, segment_ids_tensor, (all_question_span_lengths, all_column_token_lengths, all_table_token_lengths)


def _tokenize_question(question, tokenizer):
    """
    How does a question look like? Example: [['what'], ['are'], ['name'], ['of'], ['all'], ['column', 'state'], ['playing'] ... ['?']]
    What we do here is Wordpiece tokenization and adding special tokens. We further return the segment ids for the question tokens.

    @param question:
    @param tokenizer:
    @return:
    """
    question_span_lengths = [1]  # the initial value represents the length of the CLS_TOKEN in the beginning.
    all_sub_token = []

    for question_span in question:
        # remember: question-span can consist of multiple words. Example: ['column', 'state']
        sub_token = question_span
        all_sub_token.extend(sub_token)
        question_span_lengths.append(len(sub_token))

    question_tokens_with_special_chars = [tokenizer.cls_token] + all_sub_token + [tokenizer.sep_token]
    segment_ids = [SEGMENT_ID_QUESTION] * len(question_tokens_with_special_chars)

    # the additional 1 represents the SEP_TOKEN in the end.
    question_span_lengths.append(1)

    return question_tokens_with_special_chars, question_span_lengths, segment_ids


def _tokenize_column_names(column_names, tokenizer, do_sub_tokenizing=True):
    column_token_lengths = []
    all_column_tokens = []

    for column in column_names:
        if do_sub_tokenizing:
            # columns most often consists of multiple words. Here, we further tokenize them into sub-words if necessary.
            column_sub_tokens = column
        else:
            column_sub_tokens = column

        # the SEP_TOKEN needs to be packed in a list as python can only concat two lists to a new list.
        column_sub_tokens += [tokenizer.sep_token]

        all_column_tokens.extend(column_sub_tokens)
        column_token_lengths.append(len(column_sub_tokens))

    segment_ids = [SEGMENT_ID_QUESTION if tok == tokenizer.sep_token else SEGMENT_ID_SCHEMA for tok in all_column_tokens]

    return all_column_tokens, column_token_lengths, segment_ids


def _tokenize_table_names(table_names, tokenizer):
    table_token_lengths = []
    all_table_tokens = []

    for table in table_names:
        table_sub_tokens = table
        table_sub_tokens += [tokenizer.sep_token]

        all_table_tokens.extend(table_sub_tokens)
        table_token_lengths.append(len(table_sub_tokens))

    segment_ids = [SEGMENT_ID_QUESTION if tok == tokenizer.sep_token else SEGMENT_ID_SCHEMA for tok in all_table_tokens]

    return all_table_tokens, table_token_lengths, segment_ids


def _padd_input(input_ids, segment_ids, attention_mask, max_length, tokenizer):

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(tokenizer.pad_token_id)
        segment_ids.append(tokenizer.pad_token_id)

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(segment_ids) == max_length