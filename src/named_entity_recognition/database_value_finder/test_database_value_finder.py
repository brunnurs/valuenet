from unittest import TestCase

from named_entity_recognition.database_value_finder.database_value_finder import DatabaseValueFinder


class TestDatabaseValueFinder(TestCase):
    def test_find_similar_values_in_database_plural(self):
        # GIVEN
        tolerance = 0.9
        potential_values = [(candidate, tolerance) for candidate in ['Kayaking', 'names', 'professors', 'Canoeing', '1']]
        db_name = 'activity_1'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Professor', 'Rank', 'Faculty')), 0, 'Could not find "Professor" in database.')

    def test_find_similar_values_in_database_adjective(self):
        # GIVEN
        tolerance = 0.75
        potential_values = [(candidate, tolerance) for candidate in ['Canadian', 'airport', 'routes']]
        db_name = 'flight_4'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

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

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas, max_results=max_results)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

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

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Paper', 'outcome_code', 'Research_Outcomes')), 0,
                                'Could not find "Paper" in database.')

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

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('John F Kennedy International Airport', 'name', 'airports')), 0, 'Could not find "Paper" in database.')

    def test_find_similar_values_in_database___yale_sample(self):
        # GIVEN
        potential_values = [('Yale University', 0.9), ('height', 0.75), ('players', 0.75), ('college', 0.75), ('Yale', 0.75), ('University', 0.75)]
        db_name = 'baseball_1'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        # WHEN
        similar_values_db = db_value_finder.find_similar_values_in_database(potential_values)

        # THEN
        print(similar_values_db)
        self.assertGreaterEqual(similar_values_db.index(('Yale University', 'name_full', 'college')), 0, 'Could not find "Yale University" in database.')

    def test__assemble_query(self):
        # GIVEN
        columns = ['A', 'B', 'C']
        table = 'T'

        # WHEN
        query = DatabaseValueFinder._assemble_query(columns, table)

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
        query = DatabaseValueFinder._assemble_query(columns, table)

        # THEN
        self.assertEqual('SELECT TV_series.[Rating], TV_series.[18_49_Rating_Share] FROM TV_series', query)

    def test__is_similar_enough_lowercase_only(self):
        # GIVEN
        db_name = 'tracking_grants_for_research'
        db_folder = 'data/spider/original/database'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinder(db_folder, db_name, db_schemas)

        cell_value = 'Female'
        potential_value = 'female'
        tolerance = 1.0

        # WHEN
        result = db_value_finder._is_similar_enough(cell_value, potential_value, tolerance)

        # THEN
        self.assertTrue(result)

