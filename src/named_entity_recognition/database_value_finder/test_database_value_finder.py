from unittest import TestCase

from named_entity_recognition.database_value_finder.database_value_finder import DatabaseValueFinder


class TestDatabaseValueFinder(TestCase):

    def test_get_relevant_columns_with_or_without_pk_fk(self):
        # GIVEN
        db_name = 'concert_singer'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinder(db_name, db_schemas, 1)

        # WHEN
        column_table_mapping_with_pk = db_value_finder._get_relevant_columns(True)
        column_table_mapping_without_pk = db_value_finder._get_relevant_columns(False)

        # THEN
        self.assertIn('concert_ID', column_table_mapping_with_pk['concert'])
        self.assertNotIn('concert_ID', column_table_mapping_without_pk['concert'])

    def test_get_relevant_columns_text_columns_only(self):
        # GIVEN
        db_name = 'concert_singer'
        db_schemas = 'data/spider/original/tables.json'

        db_value_finder = DatabaseValueFinder(db_name, db_schemas, 1)

        # WHEN
        column_table_mapping_text_columns_only = db_value_finder._get_relevant_columns(True, True)
        column_table_mapping = db_value_finder._get_relevant_columns(True)

        # THEN
        self.assertIn('Name', column_table_mapping['stadium'])
        self.assertIn('Name', column_table_mapping_text_columns_only['stadium'])

        self.assertIn('Capacity', column_table_mapping['stadium'])
        self.assertNotIn('Capacity', column_table_mapping_text_columns_only['stadium'])
