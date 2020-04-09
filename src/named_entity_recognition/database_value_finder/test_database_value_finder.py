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
        potential_values = ['routes', 'Canadian', 'airport']
        db_name = 'flight_4'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/tables.json'

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        # self.assertGreaterEqual(similar_values_db.index(('Professor', 'Rank', 'Faculty')), 0, 'Could not find "Professor" in database.')

    def test__assemble_query(self):
        # GIVEN
        columns = ['A', 'B', 'C']
        table = 'T'

        # WHEN
        query = DatabaseValueFinder._assemble_query(columns, table)

        # THEN
        self.assertEqual('SELECT T.A, T.B, T.C FROM T', query)
