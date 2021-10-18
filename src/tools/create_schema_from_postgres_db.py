import argparse
import json

import psycopg2

from tools.create_schema_from_sqlite_db import convert_fk_index


def map_data_type(column_type: str) -> str:

    if 'text' in column_type or 'character varying' in column_type:
        return 'text'
    if 'character' in column_type:
        return 'character'
    if 'numeric' in column_type or 'double' in column_type or 'precision' in column_type or 'int' in column_type:
        return 'number'
    if 'boolean' in column_type:
        return 'boolean'

    raise ValueError(f'Unknown column type {column_type}. Please implement first.')


def load_db_meta_data(database_host, database_port, database_user, database_password, database, database_schema, create_index_statements):
    """
    Read all meta information about a database in PostgreSQL
    @param database_host:
    @param database_port:
    @param database_user:
    @param database_password:
    @param database:
    @param database_schema:
    @return:
    """
    data = {'db_id': database,
            'table_names_original': [],
            'table_names': [],
            'column_names_original': [(-1, '*')],
            'column_names': [(-1, '*')],
            'column_types': ['text'],
            'primary_keys': [],
            'foreign_keys': []}

    all_foreign_keys_temporary = []

    conn = psycopg2.connect(database=database,
                            user=database_user,
                            password=database_password,
                            host=database_host,
                            port=database_port)

    cursor = conn.cursor()
    select_tables(cursor, database_schema)

    if create_index_statements:
        print(f'In case trigram similarity (https://www.postgresql.org/docs/9.1/pgtrgm.html) is not yet supported by this database, activate it by adding this extension: CREATE EXTENSION pg_trgm;')

    for idx_table, (table_name,) in enumerate(cursor.fetchall()):
        data['table_names_original'].append(table_name)
        data['table_names'].append(table_name.lower().replace("_", ' '))

        column_cursor = conn.cursor()
        select_columns(column_cursor, table_name)

        for column_name, column_type in column_cursor.fetchall():
            data['column_names_original'].append((idx_table, column_name))
            data['column_names'].append((idx_table, column_name.lower().replace("_", " ")))

            # noinspection PyTypeChecker
            data['column_types'].append(map_data_type(column_type))

            if create_index_statements:
                if map_data_type(column_type) == 'text':
                    # build specific trigram similarity index
                    print(f'CREATE INDEX IF NOT EXISTS {table_name}_{column_name}_gist_idx ON {database_schema}.{table_name} USING GIST ({column_name} gist_trgm_ops);')
                else:
                    # build just a normal B-tree index
                    print(f'CREATE INDEX IF NOT EXISTS {table_name}_{column_name}_idx ON {database_schema}.{table_name} ({column_name});')


        foreign_key_cursor = conn.cursor()
        select_foreign_keys(foreign_key_cursor, table_name)

        all_foreign_keys_temporary.extend(
            [[(foreign_key_table, foreign_key_column), (origin_key_table, origin_key_column)] for
             foreign_key_table, foreign_key_column, origin_key_table, origin_key_column in
             foreign_key_cursor.fetchall()])

        primary_key_cursor = conn.cursor()
        select_primary_keys(primary_key_cursor, table_name)

        # also include table name, as we otherwise run into issues with duplicates given that many column appear repeatedly (as PK/FK)
        all_column_names = [f"{data['table_names_original'][t_idx]}.{column_name}" for t_idx, column_name in data['column_names_original']]
        for primary_key_name, in primary_key_cursor.fetchall():
            # noinspection PyTypeChecker
            data['primary_keys'].append(all_column_names.index(f'{table_name}.{primary_key_name}'))

    data['foreign_keys'] = all_foreign_keys_temporary
    data['foreign_keys'] = convert_fk_index(data)

    return data


def select_tables(cursor, database_schema):
    cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{database_schema}'")


def select_columns(column_cursor, table_name):
    column_cursor.execute(
        f"SELECT col.column_name, col.data_type FROM information_schema.columns col where table_name = '{table_name}';")


def select_primary_keys(primary_key_cursor, table_name):
    primary_key_cursor.execute(f"""
        SELECT c.column_name
        FROM information_schema.table_constraints tc
                 JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name)
                 JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema
            AND tc.table_name = c.table_name AND ccu.column_name = c.column_name
        WHERE constraint_type = 'PRIMARY KEY'
          and tc.table_name = '{table_name}'
        """)


def select_foreign_keys(foreign_key_cursor, table_name):
    foreign_key_cursor.execute(f"""
        SELECT tc.table_name,
               kcu.column_name,
               ccu.table_name  AS foreign_table_name,
               ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
                 JOIN information_schema.key_column_usage AS kcu
                      ON tc.constraint_name = kcu.constraint_name
                          AND tc.table_schema = kcu.table_schema
                 JOIN information_schema.constraint_column_usage AS ccu
                      ON ccu.constraint_name = tc.constraint_name
                          AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_name = '{table_name}'
        """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--database_host', default='testbed.inode.igd.fraunhofer.de', type=str)
    parser.add_argument('--database_port', default='18001', type=str)
    parser.add_argument('--database_user', default='postgres', type=str)
    parser.add_argument('--database_password', default='dummy_password', type=str)
    parser.add_argument('--database', default='oncomx_v1_0_25_small', type=str)
    parser.add_argument('--database_schema', default='oncomx_v1_0_25', type=str)

    parser.add_argument('--create_index_statements', default=False, action='store_true')

    parser.add_argument('--schema_output_file', default='tables.json', type=str)

    args = parser.parse_args()

    schema = load_db_meta_data(args.database_host,
                               args.database_port,
                               args.database_user,
                               args.database_password,
                               args.database,
                               args.database_schema,
                               args.create_index_statements)

    with open(args.schema_output_file, 'wt') as out:
        json.dump([schema], out, sort_keys=True, indent=2, separators=(',', ': '))
