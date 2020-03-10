from unittest import TestCase

from transformers import BertTokenizer

from model.encoder.input_features import encode_input, _tokenize_values


class Test(TestCase):
    def test_encode_input(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        question_spans = [[['how'], ['many'], ['table', 'student'], ['are'], ['attending'], ['english'], ['french'], ['table', 'course'], ['?']]]
        column_names = [[['count', 'number', 'many'], ['address', 'id'], ['line', '1'], ['line', '2'], ['city'], ['zip', 'postcode'], ['state', 'province', 'county'], ['country'], ['person', 'id'], ['first', 'name'], ['middle', 'name'], ['last', 'name'], ['cell', 'mobile', 'number'], ['email', 'address'], ['login', 'name'], ['password'], ['student', 'id'], ['student', 'detail'], ['course', 'id'], ['course', 'name'], ['course', 'description'], ['other', 'detail'], ['person', 'address', 'id'], ['date', 'from'], ['date', 'to'], ['registration', 'date'], ['date', 'of', 'attendance'], ['candidate', 'id'], ['candidate', 'detail'], ['qualification'], ['assessment', 'date'], ['asessment', 'outcome', 'code']]]
        table_names = [[['address'], ['people'], ['student'], ['course'], ['people', 'address'], ['student', 'course', 'registration'], ['student', 'course', 'attendance'], ['candidate'], ['candidate', 'assessment']]]
        values = [['English', 'French']]

        # WHEN
        input_ids_tensor, attention_mask_tensor, segment_ids_tensor, input_lengths = encode_input(question_spans, column_names, table_names, values, tokenizer, 512, 'cpu')

        question_span_lengths, column_token_lengths, table_token_lengths, values_lengths = input_lengths

        # THEN
        self.assertEqual(len(input_ids_tensor[0]), len(attention_mask_tensor[0]))
        self.assertEqual(len(input_ids_tensor[0]), len(segment_ids_tensor[0]))

        self.assertEqual(len(input_ids_tensor[0]), sum(question_span_lengths[0]) + sum(column_token_lengths[0]) + sum(table_token_lengths[0]) + sum(values_lengths[0]))

    def test__tokenize_values(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['English', 'French']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['english', '[SEP]', 'french', '[SEP]'], all_value_tokens)
        self.assertEqual([2, 2], value_token_lengths)
        self.assertEqual([1, 0, 1, 0], segment_ids)

    def test__tokenize_values_subword_tokenization(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['Protoporphyrinogen IX', 'dummy']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['proto', '##por', '##phy', '##rino', '##gen', 'ix', '[SEP]', 'dummy', '[SEP]'], all_value_tokens)
        self.assertEqual([7, 2], value_token_lengths)
        self.assertEqual([1, 1, 1, 1, 1, 1, 0, 1, 0], segment_ids)

    def test__tokenize_values_integer(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = [4]

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['4', '[SEP]'], all_value_tokens)
        self.assertEqual([1, 0], segment_ids)

        ids = tokenizer.convert_tokens_to_ids(all_value_tokens)

        # make sure the token is not an unknown token
        self.assertNotEqual(tokenizer.unk_token_id, ids[0])

    def test__tokenize_values_float(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['4.5']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['4', '.', '5', '[SEP]'], all_value_tokens)

        ids = tokenizer.convert_tokens_to_ids(all_value_tokens)

        # make sure the token is not an unknown token
        self.assertNotEqual(tokenizer.unk_token_id, ids[0])

    def test__tokenize_values_date(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['2002-06-21']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['2002', '-', '06', '-', '21', '[SEP]'], all_value_tokens)

        ids = tokenizer.convert_tokens_to_ids(all_value_tokens)

        # make sure no token is unknown
        self.assertListEqual([], list(filter(lambda e: e == tokenizer.unk_token_id, ids)))

    def test__tokenize_values_fuzzy_strings(self):
        # GIVEN
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        values = ['%superstar%']

        # WHEN
        all_value_tokens, value_token_lengths, segment_ids = _tokenize_values(values, tokenizer)

        # THEN
        self.assertEqual(['%', 'superstar', '%', '[SEP]'], all_value_tokens)

        ids = tokenizer.convert_tokens_to_ids(all_value_tokens)

        # make sure no token is unknown
        self.assertListEqual([], list(filter(lambda e: e == tokenizer.unk_token_id, ids)))