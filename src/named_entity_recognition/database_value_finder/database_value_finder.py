import json
import operator
from functools import reduce

import psycopg2
from more_itertools import flatten
import multiprocessing
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()

DATABASE = "cordis"
USER = "postgres"
HOST = "localhost"
PORT = "5432"
PASSWORD = "postgres"
OPTIONS = "-c search_path=unics_cordis"


class DatabaseValueFinder:
    def __init__(self, max_results=10):
        self.database_schema = self._load_schema("data/cordis/original/tables.json", DATABASE)
        self.max_results = max_results

    def find_similar_values_in_database(self, potential_values):
        matching_values = set()

        table_text_column_mapping = self._get_text_columns(self.database_schema)

        conn = psycopg2.connect(database=DATABASE, user=USER, password=PASSWORD, host=HOST, port=PORT, options=OPTIONS)

        for table, columns in table_text_column_mapping.items():
            for column in columns:
                matches = self._find_matches_in_column(table, column, potential_values, conn)
                matching_values.update(matches)

        conn.close()

        return self._top_n_results(matching_values)

    def _find_matches_in_column(self, table, column, potential_values, connection):
        query = self._assemble_query(column, table, potential_values)

        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        # r[0] is therefore valid as we query always exactly one column
        return list(map(lambda r: (r[0], column, table), rows))

    def _top_n_results(self, matching_values):
        matching_values_list = list(matching_values)
        return matching_values_list[:self.max_results]

    @staticmethod
    def _get_text_columns(database_schema):
        """
        Find all text columns in this database schema and return it in a map grouped by table.
        """
        table_columns = {}
        for idx, (table_idx, column_name) in enumerate(database_schema['column_names_original']):
            if database_schema['column_types'][idx] == 'text' and column_name != '*':
                table = database_schema['table_names_original'][table_idx]
                if table in table_columns:
                    table_columns[table].append(column_name)
                else:
                    table_columns[table] = [column_name]

        return table_columns

    @staticmethod
    def _assemble_query(column, table, potential_values):
        value_listing = reduce(lambda current, next_value: current + f", '{next_value}'", potential_values[1:], f"'{potential_values[0]}'")

        return f'SELECT {column} FROM {table} WHERE {column} IN ({value_listing})'

    @staticmethod
    def _load_schema(database_schema_path, database_name):
        with open(database_schema_path, 'r', encoding='utf-8') as json_file:
            schemas = json.load(json_file)
            for db_schema in schemas:
                if db_schema['db_id'] == database_name:
                    return db_schema

            raise ValueError(f'Schema of database "{database_name}" not found in schemas')
