import torch
from more_itertools import flatten

SEGMENT_ID_QUESTION = 0
SEGMENT_ID_SCHEMA = 1


def encode_input(question_spans, column_names, table_names, values, tokenizer, max_length_model, device):
    all_input_ids = []
    all_attention_mask = []

    all_question_span_lengths = []
    all_column_token_lengths = []
    all_table_token_lengths = []
    all_values_lengths = []

    for question, columns, tables, val in zip(question_spans, column_names, table_names, values):
        question_token_ids, question_span_lengths = _tokenize_question(question, tokenizer)
        column_token_ids, column_token_lengths = _tokenize_schema_names(columns, tokenizer)
        table_token_ids, table_token_lengths = _tokenize_schema_names(tables, tokenizer)
        value_tokens_ids, value_token_lengths = _tokenize_values(val, tokenizer)

        all_question_span_lengths.append(question_span_lengths)
        all_column_token_lengths.append(column_token_lengths)
        all_table_token_lengths.append(table_token_lengths)
        all_values_lengths.append(value_token_lengths)

        assert sum(question_span_lengths) + sum(column_token_lengths) + sum(table_token_lengths) + sum(value_token_lengths) == \
               len(question_token_ids) + len(column_token_ids) + len(table_token_ids) + len(value_tokens_ids)

        all_ids = question_token_ids + column_token_ids + table_token_ids + value_tokens_ids
        if len(all_ids) > max_length_model:
            print(f"################### ATTENTION! Example too long ({len(all_ids)}). Question-len: {len(question_token_ids)}, column-len:{len(column_token_ids)}, table-len: {len(table_token_ids)} value-len: {len(value_tokens_ids)}")
            print(question)
            print(columns)
            print(tables)
            print(values)

        # not sure here if "tokenizer.mask_token_id" or just a simple 1...
        attention_mask = [1] * len(all_ids)

        all_input_ids.append(all_ids)
        all_attention_mask.append(attention_mask)

    max_length_data = max(map(lambda ids: len(ids), all_input_ids))

    for input_ids, attention_mask in zip(all_input_ids, all_attention_mask):
        _padd_input(input_ids, attention_mask, max_length_data, tokenizer)

    input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor(all_attention_mask, dtype=torch.long).to(device)

    return input_ids_tensor, attention_mask_tensor, (all_question_span_lengths, all_column_token_lengths, all_table_token_lengths, all_values_lengths)


def _tokenize_question(question, tokenizer):
    """
    How does a question look like? Example: [['what'], ['are'], ['name'], ['of'], ['all'], ['column', 'state'], ['playing'] ... ['?']]
    What we do here is tokenization (based on the Transformer architecture we use) and adding special tokens.
    """
    question_tokenized = tokenizer(list(flatten(question)), is_split_into_words=True)
    question_tokenized_ids = question_tokenized.data['input_ids']

    question_tokenized_ids = question_tokenized_ids + [tokenizer.sep_token_id]

    question_span_lengths = [1] * len(question_tokenized_ids)

    return question_tokenized_ids, question_span_lengths


def _tokenize_schema_names(schema_elements_names, tokenizer):
    all_schema_element_length = []
    all_schema_element_ids = []

    for schema_element in schema_elements_names:
        schema_element_tokenized = tokenizer(schema_element, is_split_into_words=True)
        schema_element_ids = schema_element_tokenized.data['input_ids']

        # why the [1:]? We saw in experiments with the tokenizer that the bos_token does not appear in the second tokenization when
        # using tokenizer(text1, text_pair=text2). We therefore cut it out on purpose
        schema_element_ids_with_separator = schema_element_ids[1:] + [tokenizer.sep_token_id]

        all_schema_element_ids.extend(schema_element_ids_with_separator)
        all_schema_element_length.append(len(schema_element_ids_with_separator))

    return all_schema_element_ids, all_schema_element_length


def _tokenize_values(values, tokenizer):
    all_values_length = []
    all_values_ids = []

    for value in values:
        value = format_value(value)
        value = tokenizer(value, is_split_into_words=True)
        value_ids = value.data['input_ids']

        # why the [1:]? We saw in experiments with the tokenizer that the bos_token does not appear in the second tokenization when
        # using tokenizer(text1, text_pair=text2). We therefore cut it out on purpose
        value_ids_with_separator = value_ids[1:] + [tokenizer.sep_token_id]

        all_values_ids.extend(value_ids_with_separator)
        all_values_length.append(len(value_ids_with_separator))

    return all_values_ids, all_values_length


def _padd_input(input_ids, attention_mask, max_length, tokenizer):

    while len(input_ids) < max_length:
        # We pad the input with the official padding token of the transformer and the attention mask with a zero.
        # This is also how the tokenizer works. Example:
        # tokens_batch_3 = [['one', 'two', 'three'], ['one', 'two'], ['one']]
        # tokenized = tokenizer(tokens_batch_3, is_split_into_words=True, padding=True)
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length


def format_value(value):
    """
    This function contains heuristics to improve results, e.g. by transforming an empty string value ('') to the word empty.
    The goal is to input known values into the (transformer)-encoder, so he can learn the attention to the question.
    The heuristic in this method should stay as little as possible.
    """
    # at this point, a value needs to be a string to use the transformers tokenizing magic.
    # Any logic using numbers, needs to happen before.
    value = str(value)

    # convert empty strings to the word "empty", as the model can't handle them otherwise.
    if "".__eq__(value):
        value = 'empty'

    return value


