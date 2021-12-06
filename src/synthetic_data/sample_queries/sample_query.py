import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Union, Any

import psycopg2

from intermediate_representation.sem2sql.sem2SQL import transform, build_graph
from synthetic_data.helper import map_semql_actions_only
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
            if isinstance(query_as_semql[idx + 3], V):
                current_quadruplet = (*current_quadruplet, query_as_semql[idx + 3])
                idx += 4
            else:
                idx += 3

            column_value_quadruplets.append(current_quadruplet)
        else:
            idx += 1

    return column_value_quadruplets


def sample_table(t: T, tables: dict, generative_schema: GenerativeSchema, original_schema: dict) -> Tuple[int, str]:
    # there is a good chance that this table has been used before and is just re-mentioned
    # (e.g. multiple columns selected on the same table). In that case, don't sample a new one!
    if t.id_c in tables:
        return t.id_c, tables[t.id_c]

    table_names = generative_schema.tables

    # only sample on tables which are not yet used
    unused_tables = list(set(table_names) - set(tables.values()))

    # TODO: implement a mechanism to only sample on "close" tables, so use the schema graph to calculate distance.

    assert len(unused_tables) > 0, "we try to sample more different tables than there is in this schema"
    sampled_table = random.choice(unused_tables)

    return t.id_c, sampled_table


def sample_column(c: C, a: A, columns: dict, table_value: str, generative_schema: GenerativeSchema) -> Tuple[int, str]:
    if c.id_c in columns:
        return c.id_c, columns[c.id_c]

    # C(0) is referring to the * of the table - meaning we use all columns
    if c.id_c == 0:
        return c.id_c, '*'

    column_names = generative_schema.all_columns_of_table(table_value)

    # only sample on columns which are not yet used
    unused_columns = list(set(column_names) - set(columns.values()))

    # depending on the aggregation-type (max, min, sum, avg) we need to further filter for numeric types only
    if a.id_c in [1, 2, 4, 5]:
        unused_columns = [
            column for column in unused_columns
            if generative_schema.schema_for_column(table_value, column)['logical_datatype'] == 'number'
        ]

    assert len(unused_columns) > 0, "we try to sample more different columns than there is in this schema"
    sampled_column = random.choice(unused_columns)

    return c.id_c, sampled_column


def sample_value(v: V, values: dict, table_value: str, column_value: str, generative_schema: GenerativeSchema, db_connection: Any):
    if v.id_c in values:
        return v.id_c, values[v.id_c]

    table_meta_information = generative_schema.schema_for_table(table_value)
    column_meta_information = generative_schema.schema_for_column(table_value, column_value)

    cursor = db_connection.cursor()
    # we select a random value from the table/column we selected before.
    # Why don't we use a random selection directly on the database, e.g. by using TABLESAMPLE SYSTEM? -->
    # we want reproducible results, which requires us to set the seed for any random component
    cursor.execute(f"SELECT {table_meta_information['original_name']}.{column_meta_information['original_name']} "
                   f"FROM {table_meta_information['original_name']} LIMIT 1000")
    all_values = cursor.fetchall()

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


def replace_with_sampled_table_column_values(random_same_structure_query,columns, tables, values):

    # we replace all not-used columns/tables/values with a dummy value - to make sure we don't accidentally use the
    # original schema in our query.
    for idx in range(len(random_same_structure_query['col_set'])):
        random_same_structure_query['col_set'][idx] = 'YOU SHALL NOT SEE THIS!'

    # Then we replace the schema values with our sampled data (columns, tables, values)
    for idx, column_name in columns.items():
        random_same_structure_query['col_set'][idx] = column_name

    for idx in range(len(random_same_structure_query['table_names'])):
        random_same_structure_query['table_names'][idx] = 'YOU SHALL NOT SEE THIS!'

    for idx, table_name in tables.items():
        random_same_structure_query['table_names'][idx] = table_name

    for idx in range(len(random_same_structure_query['values'])):
        random_same_structure_query['values'][idx] = 'YOU SHALL NOT SEE THIS!'

    for idx, value in values.items():
        random_same_structure_query['values'][idx] = value

    return random_same_structure_query


def sample_query(query_type: str, spider_data: List, data_path:Path, db_connection: SimpleNamespace) -> str:

    # if query_type == 'Root1(3) Root(3) Sel(0) N(0) A(0) C(*) T(*) Filter(2) A(0) C(*) T(*) V(*)':
    #     return "SELECT biomarker_description FROM biomarker WHERE gene_symbol = 'CCL22'"
    # if query_type == 'Root1(3) Root(5) Sel(0) N(1) A(0) C(*) T(*) A(3) C(*) T(*)':
    #     return "SELECT disease.name, ae.name FROM disease INNER JOIN cancer_tissue ct ON disease.id = ct.doid INNER JOIN anatomical_entity ae ON ae.id = ct.uberon_anatomical_id WHERE ae.name = 'lung'"
    # if query_type == 'Root1(3) Root(5) Sel(0) N(0) A(3) C(*) T(*)':
    #     return "SELECT MAX(m.mutation_freq) FROM disease_mutation m INNER JOIN disease d on m.doid = d.id WHERE d.name = 'skin cancer'"
    # else:
    #     return "SELECT id from disease"
    conn = psycopg2.connect(database=db_connection.database,
                            user=db_connection.db_user,
                            password=db_connection.db_password,
                            host=db_connection.db_host,
                            port=db_connection.db_port,
                            options=db_connection.db_options)

    with open(data_path / 'original' / 'tables.json') as f:
        schemas = json.load(f)
        original_schema = schemas[0]  # we assume there is only one db-schema in this file

    generative_schema = GenerativeSchema(data_path / 'generative' / 'generative_schema.json')

    # find queries with the same structure as the query type we are looking for.
    same_structure_querys = [e for e in spider_data if map_semql_actions_only(e['rule_label']) == query_type]
    random_same_structure_query = same_structure_querys[0] # Todo: sample here from all possible ones.

    semql_structure = to_semql(random_same_structure_query['rule_label'])

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

    query_with_sampled_elements = replace_with_sampled_table_column_values(random_same_structure_query, columns, tables, values)

    transformed_sql_query = transform(query_with_sampled_elements, original_schema, query_with_sampled_elements['rule_label'])

    return transformed_sql_query[0]