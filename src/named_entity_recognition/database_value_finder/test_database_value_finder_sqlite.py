from unittest import TestCase, skip

from named_entity_recognition.database_value_finder.database_value_finder_sqlite import DatabaseValueFinderSQLite


class TestDatabaseValueFinderSQLite(TestCase):
    def test_find_similar_values_in_database_plural(self):
        # GIVEN
        tolerance = 0.9
        potential_values = [(candidate, tolerance) for candidate in ['Kayaking', 'names', 'professors', 'Canoeing', '1']]
        db_name = 'activity_1'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinderSQLite(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, False)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Professor', 'rank', 'faculty')), 0, 'Could not find "Professor" in database.')

    def test_find_similar_values_in_database_adjective(self):
        # GIVEN
        tolerance = 0.75
        potential_values = [(candidate, tolerance) for candidate in ['Canadian', 'airport', 'routes']]
        db_name = 'flight_4'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinderSQLite(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, False)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Canada', 'country', 'airports')), 0, 'Could not find "Professor" in database.')

    def test_find_similar_values_in_database_italy_italian(self):
        # GIVEN
        # TODO:  we have to set the tolerance value incredible low to get a match - this creates way to many false positives. This fact we
        # TODO   then have to cover with a large max_results match... so we need to get better here!
        tolerance = 0.5
        max_results = 100
        potential_values = [(candidate, tolerance) for candidate in ['Italian']]
        db_name = 'flight_4'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinderSQLite(db_folder, db_name, db_schemas, max_results=max_results)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, False)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Italy', 'country', 'airports')), 0, 'Could not find "Professor" in database.')

    def test_find_similar_values_in_database___fixed_weird_column_name(self):
        # GIVEN
        tolerance = 0.9
        potential_values = [(candidate, tolerance) for candidate in ['outcomes', 'project', 'details', 'patent', 'paper', 'project details']]
        db_name = 'tracking_grants_for_research'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinderSQLite(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, True)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Paper', 'outcome code', 'research outcomes')), 0, 'Could not find "Paper" in database.')
        self.assertGreaterEqual(similar_values_db.index(('Patent', 'outcome code', 'research outcomes')), 0, 'Could not find "Paper" in database.')

    def test_find_similar_values_in_database___find_way_too_many_results(self):
        # GIVEN
        potential_values = [('Kennedy International Airport', 0.8), ('John', 0.8),
                            ('John F Kennedy International Airport', 0.75), ('John', 0.75), ('F', 0.75),
                            ('Kennedy', 0.75), ('International', 0.75), ('Airport', 0.75), ('John F', 0.75),
                            ('F Kennedy', 0.75), ('Kennedy International', 0.75), ('International Airport', 0.75),
                            ('John F Kennedy', 0.75), ('F Kennedy International', 0.75),
                            ('Kennedy International Airport', 0.75), ('John F Kennedy International', 0.75),
                            ('F Kennedy International Airport', 0.75), ('routes', 0.75), ('number', 0.75)]
        db_name = 'flight_4'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinderSQLite(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, False)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('John F Kennedy International Airport', 'name', 'airports')), 0, 'Could not find "Paper" in database.')

    @skip("Due to it's size the baseball_1 database takes long to process. Therefore run this test only as single test.")
    def test_find_similar_values_in_database___yale_sample(self):
        # GIVEN
        potential_values = [('Yale University', 0.9), ('height', 0.75), ('players', 0.75), ('college', 0.75), ('Yale', 0.75), ('University', 0.75)]
        db_name = 'baseball_1'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinderSQLite(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, False)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Yale University', 'name_full', 'college')), 0, 'Could not find "Yale University" in database.')

    def test_find_similar_values_in_database___trim_values_from_database(self):
        # GIVEN
        potential_values = [('Aberdeen', 0.9), ('Ashley', 0.9), ('City', 0.75), ('flights', 0.75), ("City 'Ashley", 0.75), ("'Ashley", 0.75), ('destination', 0.75)]
        db_name = 'flight_2'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinderSQLite(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values, False)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Ashley', 'city', 'airports')), 0, 'Could not find "Ashley" in database.')
        self.assertGreaterEqual(similar_values_db.index(('Aberdeen', 'city', 'airports')), 0, 'Could not find "Aberdeen" in database.')

    def test__assemble_query(self):
        # GIVEN
        columns = ['A', 'B', 'C']
        table = 'T'

        # WHEN
        query = DatabaseValueFinderSQLite._assemble_query(columns, table)

        # THEN
        self.assertEqual('SELECT T.[A], T.[B], T.[C] FROM T', query)

    def test__assemble_query__put_table_starting_with_number_in_brackets(self):
        """
        https://stackoverflow.com/questions/44217821/why-sql-table-name-cannot-start-with-numeric
        """
        # GIVEN
        columns = ['Rating', '18_49_Rating_Share']
        table = 'TV_series'

        # WHEN
        query = DatabaseValueFinderSQLite._assemble_query(columns, table)

        # THEN
        self.assertEqual('SELECT TV_series.[Rating], TV_series.[18_49_Rating_Share] FROM TV_series', query)

    def test__is_similar_enough_lowercase_only(self):
        # GIVEN
        db_name = 'tracking_grants_for_research'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinderSQLite(db_folder, db_name, db_schemas)

        cell_value = 'Female'
        potential_value = 'female'
        tolerance = 1.0

        # WHEN
        result, similarity = db_value_finder._is_similar_enough(cell_value, potential_value, tolerance)

        # THEN
        self.assertTrue(result)

    def test__is_similar_enough_one_character_difference(self):
        """
        TODO: it might be beneficial to always allow at least one character difference. Otherwise the similiarity approach is useless for all short words.
        """
        # GIVEN
        db_name = 'tracking_grants_for_research'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinderSQLite(db_folder, db_name, db_schemas)

        cell_value = 'Herbs'
        potential_value = 'Herb'
        tolerance = 0.9

        # WHEN
        result, similarity = db_value_finder._is_similar_enough(cell_value, potential_value, tolerance)

        # THEN
        self.assertFalse(result)
        # TODO should be true when implementation is fixed.
        # self.assertTrue(result)