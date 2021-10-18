import json
import operator


def is_foreign_key(current_idx, all_foreign_keys):
    matching_foreign_keys = list(filter(lambda fk_pair: fk_pair[0] == current_idx, all_foreign_keys))
    return matching_foreign_keys != []


def is_primary_key(current_idx, all_primary_keys):
    return current_idx in all_primary_keys


def get_cleaned_column_name_by_original_column_name(column_name_original, all_original_column_names, all_cleaned_column_names):
    for idx, original in enumerate(all_original_column_names):
        name = original[1]
        if column_name_original == name:
            return all_cleaned_column_names[idx][1]

    raise ValueError(f'Could not find clean counterpart of column name {column_name_original}')


def get_cleaned_table_name_by_original_table_name(table_name_original, all_original_table_names, all_cleaned_table_names):
    idx = all_original_table_names.index(table_name_original)
    return all_cleaned_table_names[idx]


def load_schema(database_schema_path, database_name):
    with open(database_schema_path, 'r', encoding='utf-8') as json_file:
        schemas = json.load(json_file)
        for db_schema in schemas:
            if db_schema['db_id'] == database_name:
                return db_schema

        raise ValueError(f'Schema of database "{database_name}" not found in schemas')


class DatabaseValueFinder:

    def __init__(self, database_name, database_schema_path, max_results):
        self.database = database_name
        self.database_schema = load_schema(database_schema_path, database_name)
        self.max_results = max_results

    def _top_n_results(self, matching_values):
        # remember: we were dealing with a list before to avoid duplicates
        matching_values_list = list(matching_values)
        # itemgetter(1) is referring to the first element of the tuple, which is the similarity
        matching_values_list.sort(key=operator.itemgetter(1), reverse=True)
        return [(value,
                 get_cleaned_column_name_by_original_column_name(column, self.database_schema['column_names_original'],
                                                                 self.database_schema['column_names']),
                 get_cleaned_table_name_by_original_table_name(table, self.database_schema['table_names_original'],
                                                               self.database_schema['table_names']))
                for value, _, column, table in matching_values_list[:self.max_results]]

    def _get_relevant_columns(self, include_primary_keys, column_types=['text']):
        """
        To avoid too many false positives in the matches we avoid certain columns. Foreign keys are by default discarded,
        primary keys can also get discarded by setting the include_primary_keys to False.
        We can further specify, what column types we are interested in (different column types --> different similarity matching).
        To see all possible column types, have a look at create_schema_from_postgres_db.map_data_type()
        """
        table_columns = {}
        for idx, (table_idx, column_name) in enumerate(self.database_schema['column_names_original']):
            if self.database_schema['column_types'][idx] in column_types:
                if column_name != '*':
                    if not is_foreign_key(idx, self.database_schema['foreign_keys']):
                        if include_primary_keys or not is_primary_key(idx, self.database_schema['primary_keys']):
                            table = self.database_schema['table_names_original'][table_idx]
                            if table in table_columns:
                                table_columns[table].append(column_name)
                            else:
                                table_columns[table] = [column_name]

        return table_columns
