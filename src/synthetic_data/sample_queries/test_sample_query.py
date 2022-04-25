import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest import TestCase

import psycopg2

from intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, V, Root1
from synthetic_data.sample_queries.sample_query import filter_column_value_quadruplets, resolve_quadruplet, \
    find_unused_tables_closest_to_used_tables, replace_logic_names
from tools.transform_generative_schema import GenerativeSchema


class Test(TestCase):
    def test_filter_column_value_quadruplets(self):
        # Given
        query_as_semql = [Root1(3), Root(3), Sel(0), N(0), A(0), C(27), T(7), Filter(2), A(0), C(13), T(1), V(0)]

        # when
        quadruplets = filter_column_value_quadruplets(query_as_semql)

        # then
        self.assertEqual(2, len(quadruplets))

        self.assertEqual(str(quadruplets[0][0]), str(A(0)))
        self.assertEqual(str(quadruplets[0][1]), str(C(27)))
        self.assertEqual(str(quadruplets[0][2]), str(T(7)))

        self.assertEqual(str(quadruplets[1][0]), str(A(0)))
        self.assertEqual(str(quadruplets[1][1]), str(C(13)))
        self.assertEqual(str(quadruplets[1][2]), str(T(1)))
        self.assertEqual(str(quadruplets[1][3]), str(V(0)))

    def test_find_unused_tables_closest_to_used_tables(self):
        # GIVEN
        with open(Path('data/oncomx') / 'original' / 'tables.json') as f:
            original_schema = json.load(f)
            original_schema = original_schema[0]

        generative_schema = GenerativeSchema(Path('data/oncomx') / 'generative' / 'generative_schema.json')

        used_tables = [
            'disease',
            'biomarker fda test',
        ]

        unused_tables = [
            'disease mutation',
            'differential expression',
            'biomarker',
            'biomarker edrn',
            'biomarker alias',
            'biomarker fda',
            'biomarker fda test trial',
            'healthy expression',
            'stage',
            'species',
            'anatomical entity'
        ]

        # WHEN
        closest_tables = find_unused_tables_closest_to_used_tables(unused_tables, used_tables, original_schema,
                                                                   generative_schema)

        # THEN
        self.assertEqual(len(closest_tables), 4)

        self.assertIn('disease mutation', closest_tables)
        self.assertIn('differential expression', closest_tables)
        self.assertIn('biomarker fda', closest_tables)
        self.assertIn('biomarker fda test trial', closest_tables)

    def test_resolve_quadruplet(self):
        # given
        random.seed(42)

        quadruplets = filter_column_value_quadruplets(
            [Root1(3), Root(3), Sel(0), N(0), A(0), C(27), T(7), Filter(2), A(0), C(13), T(1), V(0)])

        columns = {}
        tables = {}
        values = {}

        with open(Path('data/oncomx') / 'original' / 'tables.json') as f:
            original_schema = json.load(f)
            original_schema = original_schema[0]

        generative_schema = GenerativeSchema(Path('data/oncomx') / 'generative' / 'generative_schema.json')

        db_config = SimpleNamespace(database='oncomx_v1_0_25_small',
                                    db_user='postgres',
                                    db_password='ADD_YOUR_PASSWORD_HERE',
                                    db_host='testbed.inode.igd.fraunhofer.de',
                                    db_port='18001',
                                    db_options=f"-c search_path=oncomx_v1_0_25,public")

        conn = psycopg2.connect(database=db_config.database,
                                user=db_config.db_user,
                                password=db_config.db_password,
                                host=db_config.db_host,
                                port=db_config.db_port,
                                options=db_config.db_options)

        # when
        new_columns, new_tables, new_values = resolve_quadruplet(quadruplets[0],
                                                                 columns,
                                                                 tables,
                                                                 values,
                                                                 original_schema,
                                                                 generative_schema,
                                                                 conn)

        columns[new_columns[0]] = new_columns[1]
        tables[new_tables[0]] = new_tables[1]

        # then
        self.assertEqual(7, new_tables[0])
        self.assertEqual(27, new_columns[0])
        self.assertEqual(new_values, ())

        # when
        new_columns, new_tables, new_values = resolve_quadruplet(quadruplets[1],
                                                                 columns,
                                                                 tables,
                                                                 values,
                                                                 original_schema,
                                                                 generative_schema,
                                                                 conn)

        # then
        self.assertEqual(1, new_tables[0])
        self.assertEqual(13, new_columns[0])
        self.assertEqual(0, new_values[0])

    def test_replace_logic_names(self):
        # GIVEN
        generative_schema = GenerativeSchema(Path('data/oncomx') / 'generative' / 'generative_schema.json')
        query = "SELECT T1.chromosome_pos, T1.alt_aa " \
                "FROM disease_mutation AS T1 JOIN disease AS T13 ON T1.doid = T13.id " \
                "WHERE T1.ref_aa = 'R'"

        tables = ['disease mutation', 'disease']
        columns = ['chromosome pos', 'alt aa', 'doid', 'id', 'ref aa']

        # WHEN
        replaced_query = replace_logic_names(query, tables, columns, generative_schema)

        # THEN
        self.assertEqual("SELECT T1.chromosome_position, T1.residue_mutation "
                         "FROM disease_mutation AS T1 JOIN disease AS T13 ON T1.disease_ontology_identifier = T13.disease_ontology_identifier "
                         "WHERE T1.reference_amino_acid_residue = 'R'", replaced_query)

    def test_replace_logic_names_2(self):
        # GIVEN
        generative_schema = GenerativeSchema(Path('data/oncomx') / 'generative' / 'generative_schema.json')
        query = "SELECT T1.peptide_pos FROM disease_mutation AS T1 JOIN disease_mutation_impact_prediction AS T2 ON T1.id = T2.disease_mutation_id WHERE T2.tool = 'polyphen'"

        tables = ['disease mutation', 'disease mutation impact prediction']
        columns = ['peptide pos', 'tool']

        # WHEN
        replaced_query = replace_logic_names(query, tables, columns, generative_schema)

        # THEN
        self.assertEqual("SELECT T1.amino_acid_position_mutation_protein_sequence "
                         "FROM disease_mutation AS T1 JOIN disease_mutation_impact_prediction AS T2 ON T1.id = T2.disease_mutation_id "
                         "WHERE T2.tool = 'polyphen'", replaced_query)

