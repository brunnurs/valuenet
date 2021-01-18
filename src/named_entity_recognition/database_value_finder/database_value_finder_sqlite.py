import json
import operator
import sqlite3
from functools import reduce
from pathlib import Path

from more_itertools import flatten
from textdistance import DamerauLevenshtein
import multiprocessing
from joblib import Parallel, delayed

from named_entity_recognition.database_value_finder.database_value_finder import DatabaseValueFinder

NUM_CORES = multiprocessing.cpu_count()


class DatabaseValueFinderSQLite(DatabaseValueFinder):
    def __init__(self, database_folder, database_name, database_schema_path, max_results=10):

        super().__init__(database_name, database_schema_path, max_results)

        self.database_path = Path(database_folder, database_name, database_name + '.sqlite')
        self.similarity_algorithm = DamerauLevenshtein()

        # as this thresholds are highly depending on the database specific implementation, it needs to be provided here
        self.exact_match_threshold = 1.0  # be a ware that an exact match is not case sensitive
        self.high_similarity_threshold = 0.9
        self.medium_similarity_threshold = 0.75

    def find_similar_values_in_database(self, potential_values, include_primary_keys):
        matching_values = set()

        relevant_columns = self._get_relevant_columns(include_primary_keys)

        conn = sqlite3.connect(str(self.database_path.resolve()))

        # this is necessary to avoid decoding errors with non-utf-8 content of the database
        # https://stackoverflow.com/questions/22751363/sqlite3-operationalerror-could-not-decode-to-utf-8-column
        conn.text_factory = lambda b: b.decode(errors='ignore')

        cursor = conn.cursor()

        for table, columns in relevant_columns.items():
            if columns:
                query = self._assemble_query(columns, table)

                try:
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
                except Exception as e:
                    print(f'Error while fetching data with from database {self.database} with query "{query}" Error: {e}')

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
            # for a simple comparison convert value to string and make sure to strip/trim it from whitespaces (the question tokens will never contain whitespaces!)
            cell_value = str(row[column_idx]).strip()

            for potential_value, tolerance in potential_values:
                is_similar_enough, similarity = self._is_similar_enough(cell_value, potential_value, tolerance)
                if is_similar_enough:
                    matching_value_in_database.append((cell_value, similarity))
                    potential_values_found.append(potential_value)

        return matching_value_in_database, potential_values_found

    def _is_similar_enough(self, cell_value, potential_value, tolerance):
        p = potential_value.lower()
        c = cell_value.lower()

        if tolerance < 1.0:
            similarity = self.similarity_algorithm.normalized_similarity(c, p)

            return similarity >= tolerance, similarity
        else:
            # as in these cases the result is anyway irrelevant (we require an exact match), we can also just set the
            # similarity to 0 if there is no exact match. The goal is to improve speed.
            return p == c, int(p == c)

    @staticmethod
    def fetch_data(query, cursor):
        cursor.execute(query)
        return cursor.fetchall()

    @staticmethod
    def _assemble_query(columns, table):
        # you might ask why the brackets around the column: this is necessary if a column starts with a weird character
        # like e.g. a number. And unfortunately there are some of them in the databases...
        select_clause = reduce(lambda current, next_column: current + f', {table}.[{next_column}]', columns[1:],
                               f'{table}.[{columns[0]}]')

        return f'SELECT {select_clause} FROM {table}'