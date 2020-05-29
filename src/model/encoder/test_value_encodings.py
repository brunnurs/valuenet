from unittest import TestCase

import torch
from torch.nn.utils.rnn import pad_sequence

from model.encoder.value_encodings import create_value_encodings_from_question_tokens, _find_value_in_question_tokens


class TestValueEncodings(TestCase):
    def test__create_value_encodings_from_question_tokens(self):
        # GIVEN
        question = [[['what'], ['is'], ['detail'], ['value', '5.5'], ['of'], ['table', 'student'], ['who'], ['registered'], ['most'], ['number'],
             ['of'], ['table', 'course'], ['with'], ['category'], ['monaco'], ['grand'], ['prix'], ['?']]]

        question_span_length = [[1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]]

        values = [["5.5", "Monaco Grand Prix"]]

        hidden_states = torch.rand(1, sum(question_span_length[0]) + 50, 1)

        value_hidden_states_1 = [hidden_states[0][3], hidden_states[0][4]]
        expected_averaged_value1 = torch.mean(torch.tensor(value_hidden_states_1), keepdim=True, dim=0)

        value_hidden_states_2 = [hidden_states[0][17], hidden_states[0][18], hidden_states[0][19]]
        expected_averaged_value2 = torch.mean(torch.tensor(value_hidden_states_2), keepdim=True, dim=0)

        # WHEN
        result = create_value_encodings_from_question_tokens(hidden_states, question_span_length, question, values, 'CPU')

        # THEN
        result_padded = pad_sequence(result, batch_first=True)

        self.assertEqual(len(result_padded[0]), len(values[0]))
        self.assertEqual(result_padded[0][0], expected_averaged_value1)
        self.assertEqual(result_padded[0][1], expected_averaged_value2)

    def test__find_value_in_question_tokens__fuzzy_match(self):
        # GIVEN
        question_tokens = [['find'], ['column', 'name'], ['of'], ['user'], ['whose'], ['column', 'email'], ['contain'], ['‘superstar’'], ['or'], ['‘edu’'], ['.']]
        value = '%superstar%'

        # WHEN
        result = _find_value_in_question_tokens(question_tokens, value)

        # THEN
        self.assertIsNotNone(result)
        self.assertEqual(result, range(7, 8))

    def test__find_value_in_question_tokens__float(self):
        # GIVEN
        question_tokens = [['what'], ['are'], ['distinct'], ['column', 'hometown'], ['of'], ['table', 'gymnast'],
                           ['with'], ['column', 'total', 'point'], ['more'], ['than'], ['value', '57.5'], ['?']]
        value = 57.5

        # WHEN
        result = _find_value_in_question_tokens(question_tokens, value)

        # THEN
        # we don't expect to find the value - as it's not available in the question. But we don't want it to throw an exception.
        self.assertIsNotNone(result)
        self.assertEqual(result, range(10, 11))

    def test__find_value_in_question_tokens__float_to_int(self):
        # GIVEN
        question_tokens = [['find'], ['name'], ['and'], ['column', 'gender'], ['type'], ['of'], ['table', 'dorm'], ['whose'], ['capacity'], ['is'], ['greater'], ['than'], ['value', '300'], ['or'], ['le'], ['than'], ['value', '100'], ['.']]
        value = 300.0

        # WHEN
        result = _find_value_in_question_tokens(question_tokens, value)

        # THEN
        self.assertIsNotNone(result)
        self.assertEqual(result, range(12, 13))

    def test__find_value_in_question_tokens__int(self):
        # GIVEN
        question_tokens = [['find'], ['name'], ['and'], ['column', 'gender'], ['type'], ['of'], ['table', 'dorm'], ['whose'], ['capacity'], ['is'], ['greater'], ['than'], ['value', '300'], ['or'], ['le'], ['than'], ['value', '100'], ['.']]
        value = 300

        # WHEN
        result = _find_value_in_question_tokens(question_tokens, value)

        # THEN
        self.assertIsNotNone(result)
        self.assertEqual(result, range(12, 13))

    def test__find_value_in_question_tokens__multi_word_value_at_the_end(self):
        # GIVEN
        question_tokens = [['what'], ['are'], ['column', 'name'], ['of'], ['all'], ['table', 'track'], ['that'], ['belong'], ['to'], ['rock'], ['table', 'genre'], ['and'], ['whose'], ['table', 'medium', 'type'], ['is'], ['mpeg'], ['?']]
        value = 'MPEG audio file'

        # WHEN
        result = _find_value_in_question_tokens(question_tokens, value)

        # THEN
        # we don't expect to find the value - as it's not available in the question. But we don't want it to throw an exception.
        self.assertIsNone(result)

    def test__find_value_in_question_tokens__word_to_number(self):
        # GIVEN
        question_tokens = [['show'], ['column', 'carrier'], ['of'], ['table', 'device'], ['in'], ['table', 'stock'], ['at'], ['more'], ['than'], ['one'], ['table', 'shop'], ['.']]
        value = 1

        # WHEN
        result = _find_value_in_question_tokens(question_tokens, value)

        # THEN
        # we don't expect to find the value - as it's not available in the question. But we don't want it to throw an exception.
        self.assertIsNotNone(result)
        self.assertEqual(result, range(9, 10))

    def test__find_value_in_question_tokens__quotes_type1(self):
        # GIVEN
        question_tokens = [['what'], ['is'], ['id'], ['of'], ['table', 'reviewer'], ['whose'], ['column', 'name'], ['ha'], ['substring'], ['\'mike\''], ['?']]
        value = '%Mike%'

        # WHEN
        result = _find_value_in_question_tokens(question_tokens, value)

        # THEN
        # we don't expect to find the value - as it's not available in the question. But we don't want it to throw an exception.
        self.assertIsNotNone(result)
        self.assertEqual(result, range(9, 10))

    def test__find_value_in_question_tokens__quotes_type2(self):
        # GIVEN
        question_tokens = [['what'], ['is'], ['id'], ['of'], ['table', 'reviewer'], ['whose'], ['column', 'name'], ['ha'], ['substring'], ['“mike”'], ['?']]
        value = '%Mike%'

        # WHEN
        result = _find_value_in_question_tokens(question_tokens, value)

        # THEN
        # we don't expect to find the value - as it's not available in the question. But we don't want it to throw an exception.
        self.assertIsNotNone(result)
        self.assertEqual(result, range(9, 10))

    def test__find_value_in_question_tokens__case_insensitive(self):
        # GIVEN
        question_tokens = [['How'], ['many'], ['lessons'], ['did'], ['the'], ['customer'], ['Rylan'], ['Goodwin'], ['complete'], ['?']]
        value = 'rylan'

        # WHEN
        result = _find_value_in_question_tokens(question_tokens, value)

        # THEN
        # we don't expect to find the value - as it's not available in the question. But we don't want it to throw an exception.
        self.assertIsNotNone(result)
        self.assertEqual(result, range(6, 7))