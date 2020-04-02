from unittest import TestCase

from named_entity_recognition.google_api.pre_process_ner_values import _compose_date


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
