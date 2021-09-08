from pathlib import Path
from unittest import TestCase

from pytictoc import TicToc

from config import Config
from named_entity_recognition.database_value_finder.database_value_finder_postgresql import DatabaseValueFinderPostgreSQL


class TestDatabaseValueFinderPostgreSQL(TestCase):

    def test_find_similar_values_in_database__exact_matches(self):
        # GIVEN
        potential_values = [('Belize', 1.00), ('dummy1', 0.75), ('dummy2', 0.75)]

        db_value_finder = self._initiate_db_finder()

        tic = TicToc()
        tic.tic()

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, True)

        # THEN
        tic.toc()
        print(similar_values_db)
        self.assertEqual(('Belize', 'country name', 'countries'), similar_values_db[0])

    def test_find_similar_values_in_database__missing_semicolon(self):
        # GIVEN
        potential_values = [('Bioactive compounds from meso-organisms bioconversion', 0.75)]

        db_value_finder = self._initiate_db_finder()

        tic = TicToc()
        tic.tic()

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, True)

        # THEN
        tic.toc()
        print(similar_values_db)
        self.assertEqual(("Bioactive compounds from meso-organism's bioconversion", 'title', 'topics'), similar_values_db[0])

    def test_find_similar_values_in_database__multiple_value_candidates(self):
        # GIVEN
        potential_values = [('QUANTUM', 1.0), ('ENGINEERING', 1.0), ('DEPARTMENT', 1.0), ('QUANTUM ENGINEERING DEPARTMENT', 0.7), ('Caserta', 0.7), ('many', 0.7), ('projects', 0.7), ('QUANTUM ENGINEERING', 0.7), ('ENGINEERING DEPARTMENT', 0.7), ('2015', 1.0)]

        db_value_finder = self._initiate_db_finder()

        tic = TicToc()
        tic.tic()

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, True)

        # THEN
        tic.toc()
        print(similar_values_db)
        # self.assertIn(('PROJECTS', 'department name', 'project members'), similar_values_db)
        self.assertIn(('QUANTUM ENGINEERING DEPARTMENT', 'department name', 'project members'), similar_values_db)
        self.assertIn(('Caserta', 'description', 'eu territorial units'), similar_values_db)

    def test_find_similar_values_in_database__dummy(self):
        # GIVEN
        potential_values = [('Framework Partnership Agreement', 0.75)]

        db_value_finder = self._initiate_db_finder()

        tic = TicToc()
        tic.tic()

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, True)

        # THEN
        tic.toc()
        print(similar_values_db)
        # self.assertEqual(("Bioactive compounds from meso-organism's bioconversion", 'title', 'topics'), similar_values_db[0])

    @staticmethod
    def _initiate_db_finder():
        config = {'database': 'cordis_temporary',
                  'database_host': 'testbed.inode.igd.fraunhofer.de',
                  'database_port': '18001',
                  'database_user': 'postgres',
                  'database_password': 'dummy_password', 'database_schema': 'unics_cordis'}

        base_path = Path(Config.DATA_PREFIX) / 'cordis' / 'original'
        schema_path = str(base_path / 'tables.json')

        return DatabaseValueFinderPostgreSQL(config['database'], schema_path, config, max_results=15)


