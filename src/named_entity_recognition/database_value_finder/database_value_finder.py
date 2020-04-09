import json
import sqlite3
from functools import reduce
from pathlib import Path
from textdistance import DamerauLevenshtein
import multiprocessing
from joblib import Parallel, delayed

# This parameter is critical and set by empirical experiments on the training set. Based on this value we decide between "match" and "no-match".
# We currently use the Damerau-Levenshtein string distance (normalized).
MINIMUM_SIMILARITY_THRESHOLD = 0.8

# we use multiprocessing to parallelize the search over all data.
NUM_CORES = multiprocessing.cpu_count()

class DatabaseValueFinder:
    def __init__(self, database_folder, database_name, database_schema_path):
        self.database_schema = self._load_schema(database_schema_path, database_name)
        self.database_path = Path(database_folder, database_name, database_name + '.sqlite')
        self.similarity_algorithm = DamerauLevenshtein()

    def find_similar_values_in_database(self, potential_values):
        matching_values = set()

        table_text_column_mapping = self._get_text_columns(self.database_schema)

        conn = sqlite3.connect(str(self.database_path.resolve()))
        cursor = conn.cursor()

        for table, columns in table_text_column_mapping.items():
            if columns:
                query = self._assemble_query(columns, table)

                data = self.fetch_data(query, cursor)

                for column_idx, column in enumerate(columns):
                    # as the index of the columns is equal to the way we built the query, we don't need row_factory to access the data.
                    matching_value_in_database, _ = self._find_matches_by_similarity(data, column_idx, potential_values)
                    for matching_value in matching_value_in_database:
                        matching_values.add((matching_value, column, table))

                    # We can not remove the potential value here, as we might have multiple matches over the database.

        conn.close()

        return list(matching_values)

    @staticmethod
    def fetch_data(query, cursor):
        cursor.execute(query)
        return cursor.fetchall()

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
    def _assemble_query(columns, table):
        select_clause = reduce(lambda current, next_column: current + f', {table}.{next_column}', columns[1:],
                               f'{table}.{columns[0]}')

        return f'SELECT {select_clause} FROM {table}'

    def _find_matches_by_similarity(self, data, column_idx, potential_values):
        matching_value_in_database = []
        potential_values_found = []

        for row in data:
            cell_value = row[column_idx]
            # avoid comparing None values
            if cell_value:
                for potential_value in potential_values:
                    if self.similarity_algorithm.normalized_similarity(cell_value, potential_value) >= MINIMUM_SIMILARITY_THRESHOLD:
                        matching_value_in_database.append(cell_value)
                        potential_values_found.append(potential_value)

        return matching_value_in_database, potential_values_found

    @staticmethod
    def _load_schema(database_schema_path, database_name):
        with open(database_schema_path, 'r', encoding='utf-8') as json_file:
            schemas = json.load(json_file)
            for db_schema in schemas:
                if db_schema['db_id'] == database_name:
                    return db_schema

            raise ValueError(f'Schema of database "{database_name}" not found in schemas')