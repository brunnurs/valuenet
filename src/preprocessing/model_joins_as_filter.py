import argparse
import json

from preprocessing.utils import find_table_of_star_column


def model_simple_joins_as_filter(entry):
    """
    as SemQL is not modeling any join, we need to find a way to model a situation where we join table A and B, but
    have no filter/selection on table B. To do so, we model table B as a "WHERE A.B  in = (SELECT B.ID...)" statement.
    If there are already filters, we do not only have to add one new filter for table B, but also an AND-filter on top.
    """

    sql = entry['sql']

    # this are special cases where we have multiple queries already on top-level
    if type(sql['intersect']) == dict:
        _model_joins_as_filter(sql['intersect'], entry)

    if type(sql['union']) == dict:
        _model_joins_as_filter(sql['union'], entry)

    if type(sql['except']) == dict:
        _model_joins_as_filter(sql['except'], entry)

    # here we process the general query
    _model_joins_as_filter(sql, entry)


def _model_joins_as_filter(sql, entry):
    tables_in_select = _tables_in_SELECT(sql['select'], entry['col_table'], entry)

    tables_in_filters = _tables_in_WHERE(sql['where'], entry['col_table'], entry)

    tables_in_order = _tables_in_ORDER_BY(sql['orderBy'], entry['col_table'])

    tables_used = tables_in_filters + tables_in_select + tables_in_order

    tables_in_joins = _tables_in_FROM(sql['from'])

    # Note Ursin: I'm not 100% sure why, but no all joins are fully modeled in the "FROM" clause. It seems like the
    # e.g. the tables used in the where-clause are not part of the "FROM"-clause (even though they obviously also need to be modeled as a JOIN).
    # therefore the assert here does not work. But we can still find JOIN's which are not used in select/filters.

    # assert len(set(tables_used)) <= len(set(tables_in_joins)), "Not all select/filter tables area available in the 'FROM' clause."
    #

    # the subset of tables which are not used, but in joins, is the one we need to model as filter
    joins_to_model = list(set(tables_in_joins) - set(tables_used))
    # if joins_to_model != []:
    #     print(entry['question'])
    #     print(entry['query'])
    #     for table_idx in joins_to_model:
    #         print(entry['table_names'][table_idx])
    #     print()

    for table_to_model in joins_to_model:
        table_found = False
        for condition in sql['from']['conds']:
            if type(condition) == list:
                assert condition[1] == 2, "the operator for the join is not a '='! Not implemented."

                column_first_idx = condition[2][1][1]
                column_second_idx = condition[3][1]
                table_first = entry['col_table'][column_first_idx]
                table_second = entry['col_table'][column_second_idx]

                if table_first == table_to_model:
                    join_modeled_as_filter = _model_join_as_subquery(column_first_idx, table_first, column_second_idx)
                    _add_filter_to_WHERE_clause(join_modeled_as_filter, sql)
                    table_found = True

                if table_second == table_to_model:
                    join_modeled_as_filter = _model_join_as_subquery(column_second_idx, table_second, column_first_idx)
                    _add_filter_to_WHERE_clause(join_modeled_as_filter, sql)
                    table_found = True
            else:
                print("Condition is not a list... looks like wrong data: {}".format(condition))

        if not table_found:
            print("Modeling a join failed, table could not get found in conditions. Most probably a data-issue, as e.g. in some network_1 queries.")
            print("DATABASE: {}\nQUESTION: {}\nSQL: {}\n".format(entry['db_id'], entry['question'], entry['query']))


def _model_join_as_subquery(column_to_model, table_to_model, column_connect):
    return (False, 8, (0, (0, column_connect, False), None),
                                                                {'select': (False, [(0, (0, (0, column_to_model, False), None))]),
                                                                 'from': {'table_units': [('table_unit', table_to_model)], 'conds': []},
                                                                 'where': [],
                                                                 'groupBy': [],
                                                                 'orderBy': [],
                                                                 'having': [],
                                                                 'limit': None,
                                                                 'intersect': None,
                                                                 'except': None,
                                                                 'union': None}
            , None)


def _tables_in_FROM(from_clause):
    tables_in_joins = []

    for (table_unit_type, table) in from_clause['table_units']:
        # we only consider normal joins by now - if the other "table" is a sub-query, we don't handle it.
        if table_unit_type == 'table_unit':
            tables_in_joins.append(table)

    return tables_in_joins


def _tables_in_SELECT(select_clause, column_tables_mapping, entry):
    tables_in_select = []
    # select_clause[0] is the "distinct" - flag
    for sel in select_clause[1]:
        # The "*" needs an extra handling, as it belongs not to a "normal" table.
        if sel[1][1][1] == 0:
            table_idx_of_star = find_table_of_star_column(entry, select_clause[1])
            tables_in_select.append(table_idx_of_star)
        else:
            tables_in_select.append(column_tables_mapping[sel[1][1][1]])

    return tables_in_select


def _tables_in_WHERE(where_clause, column_tables_mapping, entry):
    table_indices = []
    for condition in where_clause:
        # to avoid "or" and "and"
        if type(condition) == list:
            val_unit = condition[2]

            # table indices in condition
            if val_unit[1]:
                column_idx = val_unit[1][1]
                table_indices.append(column_tables_mapping[column_idx])

            if val_unit[2]:
                column_idx = val_unit[2][1]
                table_indices.append(column_tables_mapping[column_idx])

            # values can be sub-queries, in which case they also contain further tables.
            val1 = condition[3]
            val2 = condition[4]

            if type(val1) == dict:
                # NOTE: recursion to handle nested queries.
                _model_joins_as_filter(val1, entry)

            if type(val2) == dict:
                # NOTE: recursion to handle nested queries.
                _model_joins_as_filter(val2, entry)

    return table_indices


def _tables_in_ORDER_BY(order_by_clause, column_tables_mapping):
    table_indices = []
    if order_by_clause:
        for val_unit in order_by_clause[1]:
            col_unit1 = val_unit[1]
            col_unit2 = val_unit[2]

            if col_unit1:
                column_idx = col_unit1[1]
                table_indices.append(column_tables_mapping[column_idx])

            if col_unit2:
                column_idx = col_unit2[1]
                table_indices.append(column_tables_mapping[column_idx])

    return table_indices


def _add_filter_to_WHERE_clause(join_modeled_as_filter, sql):

    # it's important to insert the new filter in the beginning of all filters, as it needs to be a top-level filter.
    sql['where'].insert(0, join_modeled_as_filter)

    # if there have been other filters before, we need to concatenate the other filter with an AND.
    if len(sql['where']) > 1:
        sql['where'].insert(1, 'and')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    for d in data:
        model_simple_joins_as_filter(d)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
