from unittest import TestCase

from pytictoc import TicToc

from named_entity_recognition.database_value_finder.database_value_finder import DatabaseValueFinder


class TestDatabaseValueFinder(TestCase):

    def test_find_similar_values_in_database__exact_matches(self):
        # GIVEN
        potential_values = [('Belize', 1.00), ('dummy1', 0.75), ('dummy2', 0.75)]

        db_value_finder = self._initiate_db_finder()

        tic = TicToc()
        tic.tic()

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        tic.toc()
        print(similar_values_db)
        self.assertEqual(('Belize', 'country_name', 'countries'), similar_values_db[0])

    @staticmethod
    def _initiate_db_finder():
        config = {'database': 'cordis',
                  'database_host': 'localhost',
                  'database_port': '5432',
                  'database_user': 'postgres',
                  'database_password': 'postgres', 'database_schema': 'unics_cordis'}

        return DatabaseValueFinder(config['database'], 'data/cordis/original/tables.json', config)


