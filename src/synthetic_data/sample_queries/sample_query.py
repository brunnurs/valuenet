import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union, Any, Dict

import psycopg2

random.seed(42)

from intermediate_representation.sem2sql.sem2SQL import transform, build_graph
# DO NOT remove this imports! They are use by the dynamic eval() command in to_semql()
from intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, V, Root1, Action
from tools.transform_generative_schema import GenerativeSchema


def to_semql(semql_st: str):
    return [eval(x) for x in semql_st.strip().split(' ')]


def filter_column_value_quadruplets(query_as_semql: List[Action]) -> List[Tuple]:
    column_value_quadruplets = []
    idx = 0

    while idx < len(query_as_semql):
        if isinstance(query_as_semql[idx], A):
            # by definition of SemQL, columns always appear in the form A, C, T. We therefore just search for the A action
            current_quadruplet = (query_as_semql[idx], query_as_semql[idx + 1], query_as_semql[idx + 2])

            # in some cases, the A, C, T triplet is followed by a Value V
            if len(query_as_semql) > idx + 3 and isinstance(query_as_semql[idx + 3], V):
                current_quadruplet = (*current_quadruplet, query_as_semql[idx + 3])
                idx += 4
            else:
                idx += 3

            column_value_quadruplets.append(current_quadruplet)
        else:
            idx += 1

    return column_value_quadruplets


def find_unused_tables_closest_to_used_tables(unused_tables: List[str], used_tables: List[str], original_schema: dict, generative_schema: GenerativeSchema):
    original_schema_graph = build_graph(original_schema)

    distance: Dict[str, int] = {}

    # we search for each unused table the closest used table in the schema graph
    for unused_table in unused_tables:
        distance[unused_table] = 10000
        unused_table_original = generative_schema.get_original_table_name(unused_table)

        for used_table in used_tables:
            used_table_original = generative_schema.get_original_table_name(used_table)

            hops = original_schema_graph.dijkstra(used_table_original, unused_table_original)

            if len(hops) < distance[unused_table]:
                distance[unused_table] = len(hops)

    # sort by distance (ASC)
    distance_sorted: Dict[str, int] = dict(sorted(distance.items(), key=lambda item: item[1]))
    min_distance = list(distance_sorted.values())[0]

    # return all tables with a minimal distance. This is the set we then sample from.
    tables_with_min_distance = [table for table, distance in distance_sorted.items() if distance == min_distance]
    return tables_with_min_distance


def sample_table(t: T, tables: dict, generative_schema: GenerativeSchema, original_schema: dict) -> Tuple[int, str]:
    # there is a good chance that this table has been used before and is just re-mentioned
    # (e.g. multiple columns selected on the same table). In that case, don't sample a new one!
    if t.id_c in tables:
        return t.id_c, tables[t.id_c]


    table_names = generative_schema.tables

    # only sample on tables which are not yet used
    unused_tables = list(set(table_names) - set(tables.values()))

    # if it's not the first table, we want to know only the unused_tables closest to the current graph
    if len(tables) > 0:
        unused_tables = find_unused_tables_closest_to_used_tables(unused_tables,
                                                                  list(tables.values()),
                                                                  original_schema,
                                                                  generative_schema)

    assert len(unused_tables) > 0, "we try to sample more different tables than there is in this schema"
    sampled_table = random.choice(unused_tables)

    return t.id_c, sampled_table


def sample_column(c: C, a: A, columns: dict, table_value: str, generative_schema: GenerativeSchema) -> Tuple[int, str]:
    if c.id_c in columns:
        return c.id_c, columns[c.id_c]

    # C(-1) is a special case, meaning it is referring to the * of the table and we can't just sample a column
    if c.id_c == -1:
        return c.id_c, '*'

    column_names = generative_schema.all_columns_of_table(table_value)

    # only sample on columns which are not yet used
    unused_columns = list(set(column_names) - set(columns.values()))

    # depending on the aggregation-type (max, min, sum, avg) we need to further filter for numeric types only
    if a.id_c in [1, 2, 4, 5]:
        unused_columns = [
            column for column in unused_columns
            if generative_schema.get_schema_for_column(table_value, column)['logical_datatype'] == 'number'
        ]

    # C(-2) is a special case, referring to a superlative (Sup()) case. Here we can also only handle dates or numbers,
    # even though the aggregator is a A(0)
    if c.id_c == -2:
        unused_columns = [
            column for column in unused_columns
            if generative_schema.get_schema_for_column(table_value, column)['logical_datatype'] == 'number' or
               generative_schema.get_schema_for_column(table_value, column)['logical_datatype'] == 'time'
        ]

    if len(unused_columns) == 0:
        raise ValueError("we try to sample more different columns than there is in this table(s)")

    sampled_column = random.choice(unused_columns)

    return c.id_c, sampled_column


def sample_value(v: V, values: dict, table_value: str, column_value: str, generative_schema: GenerativeSchema, db_connection: Any):
    if v.id_c in values:
        return v.id_c, values[v.id_c]

    table_meta_information = generative_schema.get_schema_for_table(table_value)
    column_meta_information = generative_schema.get_schema_for_column(table_value, column_value)

    cursor = db_connection.cursor()
    # we select a random value from the table/column we selected before.
    # Why don't we use a random selection directly on the database, e.g. by using TABLESAMPLE SYSTEM? -->
    # we want reproducible results, which requires us to set the seed for any random component
    query = f"SELECT {table_meta_information['original_name']}.{column_meta_information['original_name']} " \
            f"FROM {table_meta_information['original_name']} "

    if column_meta_information['original_datatype'] == 'text':
        query += f"WHERE {table_meta_information['original_name']}.{column_meta_information['original_name']} <> '' "

    query += f"LIMIT 1000"

    cursor.execute(query)

    all_values = cursor.fetchall()

    if len(all_values) == 0:
        raise ValueError("we try to sample a value, but didn't get any value for this column which is not null")

    sampled_value = random.choice(all_values)

    return v.id_c, sampled_value[0]


def resolve_quadruplet(column_value_quadruplet: Tuple,
                       columns: dict,
                       tables: dict,
                       values: dict,
                       original_schema: dict,
                       generative_schema: GenerativeSchema,
                       db_connection: Any) -> Union[Tuple, Tuple, Tuple]:

    a: A = column_value_quadruplet[0]
    c: C = column_value_quadruplet[1]
    t: T = column_value_quadruplet[2]

    table_key, table_value = sample_table(t, tables, generative_schema, original_schema)
    column_key, column_value = sample_column(c, a, columns, table_value, generative_schema)

    if len(column_value_quadruplet) > 3:
        v: V = column_value_quadruplet[3]
        value_key, value_value = sample_value(v, values, table_value, column_value, generative_schema, db_connection)

        return (column_key, column_value), (table_key, table_value), (value_key, value_value)
    else:
        return (column_key, column_value), (table_key, table_value), ()


def assemble_spider_query_structure(columns, tables, values):
    query = {
        'col_set': [None] * len(columns.keys()),
        'table_names': [None] * len(tables.keys()),
        'values': [None] * len(values.keys())
    }

    for idx, column_name in columns.items():
        query['col_set'][idx] = column_name

    for idx, table_name in tables.items():
        query['table_names'][idx] = table_name

    for idx, value in values.items():
        query['values'][idx] = value

    return query


def replace_logic_names(sampled_query: str, tables: List[str], columns: List[str], generative_schema: GenerativeSchema):
    """
    Replace original table and column names with the "logic" names from the generative schema.
    """
    sampled_query_replaced = sampled_query

    for table in tables:
        original_table_name = generative_schema.get_original_table_name(table)
        logical_table_name = generative_schema.get_logical_table_name(table)

        sampled_query_replaced = sampled_query_replaced.replace(original_table_name, logical_table_name)

        # there is a good chance that we find the wrong column if we search in all columns - because of duplicates.
        # by focusing on a table it is still possible to find wrong columns, but far less likely
        table_schema = generative_schema.get_schema_for_table(table)
        for column in table_schema['columns']:
            if column['name'] in columns:
                sampled_query_replaced = sampled_query_replaced.replace(f".{column['original_name']}", f".{column['logical_name']}")

    return sampled_query_replaced


def sample_query(query_type: str, original_schema: dict, generative_schema: GenerativeSchema, db_connection: SimpleNamespace) -> Tuple[str, str]:

    conn = psycopg2.connect(database=db_connection.database,
                            user=db_connection.db_user,
                            password=db_connection.db_password,
                            host=db_connection.db_host,
                            port=db_connection.db_port,
                            options=db_connection.db_options)

    semql_structure = to_semql(query_type)

    column_value_quadruplets = filter_column_value_quadruplets(semql_structure)

    columns = {}
    tables = {}
    values = {}

    for column_value_quadruplet in column_value_quadruplets:
        new_columns, new_tables, new_values = resolve_quadruplet(column_value_quadruplet,
                                                                 columns,
                                                                 tables,
                                                                 values,
                                                                 original_schema,
                                                                 generative_schema,
                                                                 conn)

        columns[new_columns[0]] = new_columns[1]
        tables[new_tables[0]] = new_tables[1]

        if new_values:
            values[new_values[0]] = new_values[1]

    query_with_sampled_elements = assemble_spider_query_structure(columns, tables, values)

    transformed_sql_query = transform(query_with_sampled_elements, original_schema, query_type)[0].strip()

    sampled_query_replaced = replace_logic_names(transformed_sql_query, list(tables.values()), list(columns.values()), generative_schema)

    return transformed_sql_query, sampled_query_replaced
