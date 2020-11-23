import json

################ Notes to myself ############################
# While this piece of code works to remove tables from the schema, it does NOT adapt the training samples!
# This is quite critical though as the SQL-tree in the  training samples contain references to column/tables by index. So if we change this indices,
# we also need to change the one in the training samples.
# As this is a shitload of work, I go back to the initial approach of just not sub-tokenizing the schemas which are too long.
# So this CODE IS CURRENTLY NOT USED!
#############################################################

def _get_columns_of_table(table_to_remove):
    columns_to_remove = []
    for idx, (table_idx, column_name) in enumerate(baseball_schema['column_names']):
        if table_idx == table_to_remove:
            columns_to_remove.append(idx)

    return columns_to_remove


def _remove_all_instances_of_relations(idx_from, idx_to):
    del baseball_schema['foreign_keys'][idx_from:idx_to + 1]


def _remove_all_instances_of_columns(idx_from, idx_to):
    del baseball_schema['column_names'][idx_from:idx_to + 1]
    del baseball_schema['column_names_original'][idx_from:idx_to + 1]
    del baseball_schema['column_types'][idx_from:idx_to + 1]


def _remove_all_instances_of_table(table_idx):
    del baseball_schema['table_names'][table_idx]
    del baseball_schema['table_names_original'][table_idx]


def _find_PK_FK_relations(columns_idx):
    relations_idx = []
    for idx, (pk, fk) in enumerate(baseball_schema['foreign_keys']):
        if pk in columns_idx or fk in columns_idx:
            relations_idx.append(idx)

    return relations_idx


def _adapt_foreign_keys_indices(columns_idx):
    # we only need to restore order of indices which are larger than the largest index we remove. Example:
    # we removed index 103 - 121. All indices < 103 stay the same and all relations to 103 - 121 we just removed before.
    # so it's sufficient to adapt all indices > 121.
    max_removed_column_idx = columns_idx[-1]
    n_columns_removed = len(columns_idx)
    for idx, (pk, fk) in enumerate(baseball_schema['foreign_keys']):
        if pk > max_removed_column_idx:
            baseball_schema['foreign_keys'][idx][0] = baseball_schema['foreign_keys'][idx][0] - n_columns_removed

        if fk > max_removed_column_idx:
            baseball_schema['foreign_keys'][idx][1] = baseball_schema['foreign_keys'][idx][1] - n_columns_removed


def _adapt_column_table_indices(table_idx, column_type):
    for idx, (table_idx_of_column, _) in enumerate(baseball_schema[column_type]):
        # for each column which appears after the table we removed, we need to adapt the table index. As we removed
        # exactly one column, it is sufficient to just decrement the the index by 1.
        if table_idx_of_column > table_idx:
            baseball_schema[column_type][idx][0] = table_idx_of_column - 1


def remove_table_with_columns_and_relations(table_name):
    """
    Remove all information about this table from the schema, including column, column type and foreign keys.
    After removing, we need to adapt all foreign keys and columns which are based on an index of a table/column.
    """
    table_idx = baseball_schema['table_names'].index(table_name)
    _remove_all_instances_of_table(table_idx)

    columns_idx = _get_columns_of_table(table_idx)
    relations_idx = _find_PK_FK_relations(columns_idx)

    # be aware that the following two methods only work if all columns are consecutive.
    _remove_all_instances_of_columns(columns_idx[0], columns_idx[-1])
    _remove_all_instances_of_relations(relations_idx[0], relations_idx[-1])

    _adapt_column_table_indices(table_idx, 'column_names')
    _adapt_column_table_indices(table_idx, 'column_names_original')
    _adapt_foreign_keys_indices(columns_idx)


if __name__ == '__main__':
    with open('data/spider/tables.json', 'r', encoding='utf8') as f:
        tables = json.load(f)

    baseball_schema = next(filter(lambda t: t['db_id'] == 'baseball_1', tables), None)

    remove_table_with_columns_and_relations('fielding')
    remove_table_with_columns_and_relations('fielding outfield')
    remove_table_with_columns_and_relations('fielding postseason')

    with open('data/spider/tables_manipulated.json', 'w', encoding='utf-8') as f:
        json.dump(tables, f)