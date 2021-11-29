import argparse
import json
from pathlib import Path
from typing import List


def transform(original_schema_path: Path, new_schema_path: Path, tables_of_interest:List[str]):
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
            "columns": []
        }

        for column_idx, column in enumerate(original_schema['column_names_original']):
            column_table_idx = column[0]

            # we loop over all columns, but only care about the columns of the current table.
            if table_idx == column_table_idx:
                column = {
                    "name": original_schema['column_names'][column_idx][1],
                    "original_name": column[1],
                    "logical_datatype": original_schema['column_types'][column_idx],
                    "original_datatype": original_schema['column_types'][column_idx]
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
        'biomarker_fda_test',
        'differential_expression',
        'biomarker'
    ]

    transform(args.original_schema, args.new_schema, tables)
