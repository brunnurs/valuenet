from unittest import TestCase

from named_entity_recognition.database_value_finder.database_value_finder import DatabaseValueFinder


class TestDatabaseValueFinder(TestCase):
    def test_find_similar_values_in_database_plural(self):
        # GIVEN
        potential_values = ['Kayaking', 'names', 'professors', 'Canoeing', '1']
        db_name = 'activity_1'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/tables.json'

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Professor', 'Rank', 'Faculty')), 0, 'Could not find "Professor" in database.')

    def test_find_similar_values_in_database_adjective(self):
        # GIVEN
        potential_values = ['Canadian', 'airport', 'routes']
        db_name = 'flight_4'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/tables.json'

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Canada', 'country', 'airports')), 0, 'Could not find "Professor" in database.')

    def test_find_similar_values_in_database_italy_italian(self):
        # TODO 'Italian' and 'Italy' is just too far away for the string similarity. What can we do?
        # GIVEN
        potential_values = ['Italian']
        db_name = 'flight_4'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/tables.json'

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Italy', 'country', 'airports')), 0, 'Could not find "Professor" in database.')

    def test_find_similar_values_in_database___fixed_weird_column_name(self):
        # GIVEN
        potential_values = ['outcomes', 'project', 'details', 'patent', 'paper', 'project details']
        db_name = 'tracking_grants_for_research'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/tables.json'

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Paper', 'outcome_code', 'Research_Outcomes')), 0,
                                'Could not find "Paper" in database.')

    def test__assemble_query(self):
        # GIVEN
        columns = ['A', 'B', 'C']
        table = 'T'

        # WHEN
        query = DatabaseValueFinder._assemble_query(columns, table)

        # THEN
        self.assertEqual('SELECT T.A, T.B, T.C FROM T', query)
