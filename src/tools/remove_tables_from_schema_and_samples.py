import json


def _get_columns_of_table(schema, table_to_remove):
    columns_to_remove = []
    for idx, (table_idx, column_name) in enumerate(schema['column_names']):
        if table_idx == table_to_remove:
            columns_to_remove.append(idx)

    return columns_to_remove


def _remove_all_instances_of_relations(schema, idx_from, idx_to):
    del schema['foreign_keys'][idx_from:idx_to + 1]


def _remove_all_instances_of_columns(schema, idx_from, idx_to):
    del schema['column_names'][idx_from:idx_to + 1]
    del schema['column_names_original'][idx_from:idx_to + 1]
    del schema['column_types'][idx_from:idx_to + 1]


def _remove_all_instances_of_table(schema, table_idx):
    del schema['table_names'][table_idx]
    del schema['table_names_original'][table_idx]


def _find_PK_FK_relations(schema, columns_idx):
    relations_idx = []
    for idx, (pk, fk) in enumerate(schema['foreign_keys']):
        if pk in columns_idx or fk in columns_idx:
            relations_idx.append(idx)

    return relations_idx


def _adapt_foreign_keys_indices(schema, columns_idx):
    # we only need to restore order of indices which are larger than the largest index we remove. Example:
    # we removed index 103 - 121. All indices < 103 stay the same and all relations to 103 - 121 we just removed before.
    # so it's sufficient to adapt all indices > 121.
    max_removed_column_idx = columns_idx[-1]
    n_columns_removed = len(columns_idx)
    for idx, (pk, fk) in enumerate(schema['foreign_keys']):
        if pk > max_removed_column_idx:
            schema['foreign_keys'][idx][0] = schema['foreign_keys'][idx][0] - n_columns_removed

        if fk > max_removed_column_idx:
            schema['foreign_keys'][idx][1] = schema['foreign_keys'][idx][1] - n_columns_removed


def _adapt_column_table_indices(schema, table_idx, column_type):
    for idx, (table_idx_of_column, _) in enumerate(schema[column_type]):
        # for each column which appears after the table we removed, we need to adapt the table index. As we removed
        # exactly one column, it is sufficient to just decrement the the index by 1.
        if table_idx_of_column > table_idx:
            schema[column_type][idx][0] = table_idx_of_column - 1


def _adapt_indices(sql_structure, table_idx,  min_index, reduction, columns_indices_which_should_not_appear):
    if isinstance(sql_structure, dict):
        for k, v in sql_structure.items():
            _adapt_indices(v, table_idx, min_index, reduction, columns_indices_which_should_not_appear)
    # per definition of the sql structure the indices to columns are always in a list - therefore we only need to check and adapt here.
    elif isinstance(sql_structure, list):
        for idx, item in enumerate(sql_structure):
            # "table_unit" is kind of a special case: we also need to adapt the table references!
            if item == 'table_unit':
                # we know that the next element will be the index of the table
                if sql_structure[idx + 1] > table_idx:
                    # if the index is above the table-idx we just removed, we need to decrement it.
                    sql_structure[idx + 1] = sql_structure[idx + 1] - 1

            if isinstance(item, int):
                if item >= min_index:
                    sql_structure[idx] = item - reduction
                    print(f'We found a reference: {item}. The reference is  > min-index ({min_index}) and therefore we reduced it by {reduction}')

                if item in columns_indices_which_should_not_appear:
                    raise ValueError('There should be no more references to this table! Otherwise we are not allowed to remove it!')

            elif isinstance(item, list) or isinstance(item, dict):
                _adapt_indices(item, table_idx, min_index, reduction, columns_indices_which_should_not_appear)


def _adapt_sql_structure_in_samples(table_idx, columns_idx, samples, db_id):
    for sample in samples:
        if sample['db_id'] == db_id:
            print(f"Start adapting references to schema {db_id} in sample: {sample['question']}")
            min_index_to_adapt = columns_idx[-1] + 1
            reduction = len(columns_idx)
            _adapt_indices(sample['sql'], table_idx, min_index_to_adapt, reduction, columns_idx)

    print(f"Finished adapting references to schema {db_id} in all samples.")


def remove_table_from_schema_and_samples(table_name, schema, samples, db_id):
    """
    Remove all information about this table from the schema, including column, column type and foreign keys.
    After removing, we need to adapt all foreign keys and columns which are based on an index of a table/column.
    """
    table_idx = schema['table_names'].index(table_name)
    _remove_all_instances_of_table(schema, table_idx)

    columns_idx = _get_columns_of_table(schema, table_idx)
    relations_idx = _find_PK_FK_relations(schema, columns_idx)

    # be aware that the following two methods only work if all columns are consecutive.
    _remove_all_instances_of_columns(schema, columns_idx[0], columns_idx[-1])
    if relations_idx:
        _remove_all_instances_of_relations(schema, relations_idx[0], relations_idx[-1])
    else:
        print(f"No relations found for table {table_name}")

    _adapt_column_table_indices(schema, table_idx, 'column_names')
    _adapt_column_table_indices(schema, table_idx, 'column_names_original')
    _adapt_foreign_keys_indices(schema, columns_idx)

    _adapt_sql_structure_in_samples(table_idx, columns_idx, samples, db_id)


if __name__ == '__main__':
    ##### ATTENTION with this Script! ############
    # It is not perfectly tested and quite dangerous. So check if the data have been adapted as you expect it after running it.
    # You should also NEVER use it for low indices (< 30 maybe?) as it searches for column-indices in the training samples just by value -
    # therefore if your column-idx is 3, it will most probably also find all kind of other values with 3 and adapt it!
    ###########################################


    with open('data/spider/original/tables.json', 'r', encoding='utf8') as f:
        tables = json.load(f)

    with open('data/spider/original/train_spider.json', 'r', encoding='utf8') as f:
        samples = json.load(f)

    db_id = 'baseball_1'

    _baseball_schema = next(filter(lambda t: t['db_id'] == db_id, tables))

    # remove_table_from_schema_and_samples('team half', _baseball_schema, samples, db_id)
    # remove_table_from_schema_and_samples('pitching postseason', _baseball_schema, samples, db_id)
    # remove_table_from_schema_and_samples('pitching', _baseball_schema, samples, db_id)
    # remove_table_from_schema_and_samples('manager half', _baseball_schema, samples, db_id)
    # remove_table_from_schema_and_samples('manager', _baseball_schema, samples, db_id)
    # remove_table_from_schema_and_samples('fielding postseason', _baseball_schema, samples, db_id)
    # remove_table_from_schema_and_samples('fielding outfield', _baseball_schema, samples, db_id)
    remove_table_from_schema_and_samples('fielding', _baseball_schema, samples, db_id)

    with open('data/spider/original/tables.json', 'w', encoding='utf-8') as f:
        json.dump(tables, f, indent=4)

    with open('data/spider/original/train_spider.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)
