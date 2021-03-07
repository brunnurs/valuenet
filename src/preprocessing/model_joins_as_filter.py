import argparse
import json

from intermediate_representation.sem2sql.infer_from_clause import generate_path_by_graph
from intermediate_representation.sem2sql.sem2SQL import build_graph
from preprocessing.utils import find_table_of_star_column
from spider.spider_utils import load_schema


def model_simple_joins_as_filter(entry, schema):
    """
    as SemQL is not modeling any join, we need to find a way to model a situation where we join table A and B, but
    have no filter/selection on table B. To do so, we model table B as a "WHERE A.B  in = (SELECT B.ID...)" statement.
    If there are already filters, we do not only have to add one new filter for table B, but also an AND-filter on top.
    """

    sql = entry['sql']

    # this are special cases where we have multiple queries already on top-level
    if type(sql['intersect']) == dict:
        _model_joins_as_filter(sql['intersect'], entry, schema)

    if type(sql['union']) == dict:
        _model_joins_as_filter(sql['union'], entry, schema)

    if type(sql['except']) == dict:
        _model_joins_as_filter(sql['except'], entry, schema)

    # here we process the general query
    _model_joins_as_filter(sql, entry, schema)


def _model_joins_as_filter(sql, entry, schema):
    tables_in_select = _tables_in_SELECT(sql['select'], entry['col_table'], entry)

    tables_in_filters = _tables_in_WHERE(sql['where'], entry['col_table'], entry, schema)

    tables_in_order = _tables_in_ORDER_BY(sql['orderBy'], entry['col_table'])

    tables_used = tables_in_filters + tables_in_select + tables_in_order

    # with "tables_used" we know all the tables a user will most probably mention in a question. What we are looking for
    # now are the tables he also mentioned, but will be part of a join (which is not a thing for SemQL so we will model them as Filters)
    # But some tables we shouldn't model as filters, as they are simple "in-between" tables and will anyways be embedded into the query
    # in the post processing. Therefore we extend the "tables_used" with this in-between tables, the same way as the post-processing does it.
    tables_used_total = _complement_with_in_between_tables(schema, set(tables_used))

    # this "insurance" here is necessary for all cases in which the schema is not connected (foreign keys) as it should.
    # in that case the graph will not find a way to connect the tables and therefore have no tables in the end. If that's the case
    # we just don't care about the in-between tables.
    if len(tables_used_total) >= len(set(tables_used)):
        tables_used = tables_used_total
    else:
        print(f"Seems like database '{entry['db_id']}' is missing foreign keys which would be necessary for query: '{entry['query']}'")
        print()

    # now we want to know if there is a table missing in our "used" list, so some inherent filter based on joins only.
    tables_in_from = _tables_in_FROM(sql['from'])

    # Note Ursin: I'm not 100% sure why, but no all joins are fully modeled in the "FROM" clause. It seems like the
    # e.g. the tables used in the where-clause are not part of the "FROM"-clause (even though they obviously also need to be modeled as a JOIN).
    # therefore the assert here does not work. But we can still find JOIN's which are not used in select/filters.

    # assert len(set(tables_used)) <= len(set(tables_in_joins)), "Not all select/filter tables area available in the 'FROM' clause."
    #

    # the subset of tables which are not used, but in joins, is the one we need to model as filter
    joins_to_model = list(set(tables_in_from) - set(tables_used))
    # if joins_to_model != []:
    #     print(entry['question'])
    #     print(entry['query'])
    #     for table_idx in joins_to_model:
    #         print(entry['table_names'][table_idx])
    #     print()

    if joins_to_model:
        print(f"Model joins as extra filter for the following tables: {', '.join([entry['table_names'][t] for t in joins_to_model])}")
        print(f"Question: {entry['question']}")
        print(f"Query: '{entry['query']}'")
        print(f"Database: '{entry['db_id']}'")
        print()

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
            elif condition == 'or' or condition == 'and':
                # we are only interested in the actual conditions, so we pass here
                pass
            else:
                print("Condition is not a list... looks like wrong data: {}".format(condition))

        if not table_found:
            print("Modeling a join failed, table could not get found in conditions. Most probably a data-issue, as e.g. in some network_1 queries.")
            print("DATABASE: {}\nQUESTION: {}\nSQL: {}\n".format(entry['db_id'], entry['question'], entry['query']))


def _complement_with_in_between_tables(schema, tables_used):

    # special case: if there is only one table we can not find in-between table. Even worse, we will return 0 tables.
    if len(tables_used) == 1:
        return tables_used

    # the graph uses the original table name, so we have to switch from the index to that one and back in the end of the method.    
    table_names = {schema['table_names_original'][table]: f'T{idx + 1}' for idx, table in enumerate(tables_used)}

    graph = build_graph(schema)
    join_clauses, _ = generate_path_by_graph(graph, table_names, list(table_names.keys()))

    # for mor information how the join-clause tuple looks, see test_infer_from_clause.py
    tables_used_including_in_between_tables = [table1 for table1, _, _, _, _, _ in join_clauses]

    # most tables appear twice when we do this, but there is some exception case with the first and the last table in a join.
    # We de-dup with the "set()" anyway, so this is totally fine.
    tables_used_including_in_between_tables.extend([table2 for _, _, table2, _, _, _ in join_clauses])

    tables_used_deduplicated = list(set(tables_used_including_in_between_tables))

    # back to indices...
    return [schema['table_names_original'].index(table) for table in tables_used_deduplicated]


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


def _tables_in_WHERE(where_clause, column_tables_mapping, entry, schema):
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
                _model_joins_as_filter(val1, entry, schema)

            if type(val2) == dict:
                # NOTE: recursion to handle nested queries.
                _model_joins_as_filter(val2, entry, schema)

    return table_indices


def _tables_in_ORDER_BY(order_by_clause, column_tables_mapping):
    table_indices = []
    if order_by_clause:
        for val_unit in order_by_clause[1]:
            col_unit1 = val_unit[1]
            col_unit2 = val_unit[2]

            if col_unit1:
                column_idx = col_unit1[1]
                # this if is referring to a special case where column-idx is referring to "*", which is not a table, but
                # a group over all attributes e.g. GROUP BY count(*).
                if column_tables_mapping[column_idx] != -1:
                    table_indices.append(column_tables_mapping[column_idx])

            if col_unit2:
                column_idx = col_unit2[1]
                # this if is referring to a special case where column-idx is referring to "*", which is not a table, but
                # a group over all attributes e.g. GROUP BY count(*).
                if column_tables_mapping[column_idx] != -1:
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
    arg_parser.add_argument('--table_path', type=str, help='Schema information', required=True)
    arg_parser.add_argument('--output', type=str, help='output data')
    args = arg_parser.parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    _, schema_dict = load_schema(args.table_path)

    for d in data:
        schema_for_db = schema_dict[d['db_id']]
        model_simple_joins_as_filter(d, schema_for_db)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
