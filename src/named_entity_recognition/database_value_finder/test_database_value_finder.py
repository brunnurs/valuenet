from unittest import TestCase

from named_entity_recognition.database_value_finder.database_value_finder import DatabaseValueFinder


class TestDatabaseValueFinder(TestCase):
    def test_find_similar_values_in_database__exact_matches(self):
        # GIVEN
        potential_values = [('Belize', 0.75), ('dummy1', 0.75), ('dummy2', 0.75)]

        db_value_finder = DatabaseValueFinder('cordis', 'data/cordis/original/tables.json')

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        self.assertEqual(('Belize', 'country_name', 'countries'), similar_values_db[0])

