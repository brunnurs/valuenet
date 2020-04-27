import json
import operator
import sqlite3
from functools import reduce
from pathlib import Path
import pytictoc

from more_itertools import flatten
from textdistance import DamerauLevenshtein
import multiprocessing
from joblib import Parallel, delayed

NUM_CORES = multiprocessing.cpu_count()


class DatabaseValueFinder:
    def __init__(self, database_folder, database_name, database_schema_path, max_results=10):
        self.database = database_name
        self.database_schema = self._load_schema(database_schema_path, database_name)
        self.database_path = Path(database_folder, database_name, database_name + '.sqlite')
        self.similarity_algorithm = DamerauLevenshtein()
        self.max_results = max_results

    def find_similar_values_in_database(self, potential_values):
        matching_values = set()

        table_text_column_mapping = self._get_text_columns(self.database_schema)

        conn = sqlite3.connect(str(self.database_path.resolve()))
        cursor = conn.cursor()

        for table, columns in table_text_column_mapping.items():
            if columns:
                query = self._assemble_query(columns, table)

                data = self.fetch_data(query, cursor)

                # The overhead of parallelization only helps after a certain size of data. Example: a table with ~ 300k entries and 4 columns takes ~20s with a single core.
                # By using all 12 virtual cores we get down to ~12s. But the table has only 60k entries and 4 columns, the overhead of parallelization is larger than calculating
                # everything on a single core (~3.8s vs. ~4.1s)
                if len(data) > 80000:
                    matches = Parallel(n_jobs=NUM_CORES)(delayed(self._find_matches_in_column)(table, column, column_idx, data, potential_values) for column_idx, column in enumerate(columns))
                    print(f'Parallelization activated as table has {len(data)} rows.')
                else:
                    matches = [self._find_matches_in_column(table, column, column_idx, data, potential_values) for column_idx, column in enumerate(columns)]

                matching_values.update(flatten(matches))

        conn.close()

        return self._top_n_results(matching_values)

    def _find_matches_in_column(self, table, column, column_idx, data, potential_values):
        # as the index of the columns is equal to the way we built the query, we don't need row_factory to access the data.
        matching_value_in_database, _ = self._find_matches_by_similarity(data, column_idx, potential_values)

        return list(map(lambda value_similarity: (value_similarity[0], value_similarity[1], column, table), matching_value_in_database))

    def _find_matches_by_similarity(self, data, column_idx, potential_values):
        matching_value_in_database = []
        potential_values_found = []

        for row in data:
            cell_value = row[column_idx]
            # avoid comparing None values
            if cell_value and isinstance(cell_value, str):
                for potential_value, tolerance in potential_values:
                    is_similar_enough, similarity = self._is_similar_enough(cell_value, potential_value, tolerance)
                    if is_similar_enough:
                        matching_value_in_database.append((cell_value, similarity))
                        potential_values_found.append(potential_value)

        return matching_value_in_database, potential_values_found

    def _is_similar_enough(self, cell_value, potential_value, tolerance):
        p = potential_value.lower()
        c = cell_value.lower()

        similarity = self.similarity_algorithm.normalized_similarity(c, p)

        return similarity >= tolerance, similarity

    def _top_n_results(self, matching_values):
        # remember: we were dealing with a list before to avoid duplicates
        matching_values_list = list(matching_values)
        # itemgetter(1) is referring to the first element of the tuple, which is the similarity
        matching_values_list.sort(key=operator.itemgetter(1), reverse=True)
        return [(value, column, table) for value, _, column, table in matching_values_list[:self.max_results]]

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
        # you might ask why the brackets around the column: this is necessary if a column starts with a weird character
        # like e.g. a number. And unfortunately there are some of them in the databases...
        select_clause = reduce(lambda current, next_column: current + f', {table}.[{next_column}]', columns[1:],
                               f'{table}.[{columns[0]}]')

        return f'SELECT {select_clause} FROM {table}'

    @staticmethod
    def _load_schema(database_schema_path, database_name):
        with open(database_schema_path, 'r', encoding='utf-8') as json_file:
            schemas = json.load(json_file)
            for db_schema in schemas:
                if db_schema['db_id'] == database_name:
                    return db_schema

            raise ValueError(f'Schema of database "{database_name}" not found in schemas')