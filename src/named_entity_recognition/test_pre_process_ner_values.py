from unittest import TestCase

from named_entity_recognition.pre_process_ner_values import _compose_date, \
    _build_ngrams


class Test(TestCase):
    def test__compose_date(self):
        # GIVEN
        full_date = {
            'name': "July 5th, 2009",
            'type': "DATE",
            'metadata': {
                'year': "2009",
                'day': "5",
                'month': "7"
            }
        }

        expected = '2009-07-05'

        # WHEN
        actual = _compose_date(full_date)

        # THEN
        self.assertEqual(expected, actual)

    def test__compose_date__year(self):
        # GIVEN
        full_date = {
            'name': "2009",
            'type': "DATE",
            'metadata': {
                'year': "2009",
            }
        }

        expected = '2009'

        # WHEN
        actual = _compose_date(full_date)

        # THEN
        self.assertEqual(expected, actual)

    def test__compose_date__month_day(self):
        # GIVEN
        full_date = {
            'name': "July 5th",
            'type': "DATE",
            'metadata': {
                'day': "5",
                'month': "7"
            }
        }

        expected = '07-05'

        # WHEN
        actual = _compose_date(full_date)

        # THEN
        self.assertEqual(expected, actual)

    def test__compose_date__year_month(self):
        # GIVEN
        full_date = {
            'name': "July, 2009",
            'type': "DATE",
            'metadata': {
                'year': "2009",
                'month': "7"
            }
        }

        expected = '2009-07'

        # WHEN
        actual = _compose_date(full_date)

        # THEN
        self.assertEqual(expected, actual)

    def test__build_ngrams(self):
        # GIVEN
        multi_word_input = "Peter Smith"

        # WHEN
        combinations = _build_ngrams(multi_word_input)

        # THEN (assertCountEqual() is verifying list content is the same no matter the order - an outright misleading name)
        self.assertCountEqual(['Peter', 'Smith', 'Peter Smith'], combinations)

    def test__build_simplified_ngrams_long_sequence(self):
        # GIVEN
        multi_word_input = "Peter Martin Smith Müller"

        # WHEN
        combinations = _build_ngrams(multi_word_input)

        print(combinations)
        # # THEN (assertCountEqual() is verifying list content is the same no matter the order - an outright misleading name)
        # self.assertCountEqual(['Peter Martin', 'Martin Smith', 'Martin', 'Peter Martin Smith'], combinations)
