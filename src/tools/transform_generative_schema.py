import argparse
import json
from pathlib import Path
from typing import List, Dict


class GenerativeSchema:
    def __init__(self, generative_schema_path: Path) -> None:
        with open(generative_schema_path) as f:
            self.schema = json.load(f)

    @property
    def tables(self) -> List[str]:
        return [table['name'] for table in self.schema]

    def get_schema_for_table(self, table: str) -> Dict:
        return [t for t in self.schema if t['name'] == table][0]

    def get_schema_for_column(self, table: str, column: str) -> Dict:
        table_schema = self.get_schema_for_table(table)
        return [c for c in table_schema['columns'] if c['name'] == column][0]

    def all_columns_of_table(self, table: str) -> List[str]:
        table_schema = self.get_schema_for_table(table)
        return [column['name'] for column in table_schema['columns']]

    def get_original_table_name(self, table: str) -> str:
        return self.get_schema_for_table(table)['original_name']

    def get_logical_table_name(self, table: str) -> str:
        return self.get_schema_for_table(table)['logical_name']


def transform(original_schema_path: Path, new_schema_path: Path, tables_of_interest: List[str]):
    with open(original_schema_path) as f:
        original_schema = json.load(f)

    # assuming there is only one schema per file
    original_schema = original_schema[0]

    new_schema = []
    for table_idx, table_original in enumerate(original_schema['table_names_original']):

        # we don't consider all tables but just the one of interest. This normally does not include simple
        # connection tables.
        if table_original not in tables_of_interest:
            continue

        table = {
            "name": original_schema['table_names'][table_idx],
            "original_name": table_original,
            "logical_name": table_original,
            "columns": []
        }

        for column_idx, column in enumerate(original_schema['column_names_original']):
            column_table_idx = column[0]

            # we loop over all columns, but only care about the columns of the current table.
            if table_idx == column_table_idx:
                column = {
                    "name": original_schema['column_names'][column_idx][1],
                    "original_name": column[1],
                    "logical_name": column[1],
                    "original_datatype": original_schema['column_types'][column_idx],
                    "logical_datatype": original_schema['column_types'][column_idx],
                }

                table["columns"].append(column)

        new_schema.append(table)

        # TODO: do we need to add relationship (PK/FK) information?

        with open(new_schema_path, 'wt') as out:
            json.dump(new_schema, out, indent=2, )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--original_schema', default='data/oncomx/original/tables.json', type=str)
    parser.add_argument('--new_schema', default='data/oncomx/generative/generative_schema.json', type=str)

    args = parser.parse_args()

    # we don't consider all tables but just the one of interest. This normally does not include simple connection tables.
    tables = [
        'disease',
        'disease_mutation',
        'anatomical_entity',
        'disease_mutation_impact_prediction'
    ]

    transform(args.original_schema, args.new_schema, tables)
