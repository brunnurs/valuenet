import json
from types import SimpleNamespace
from typing import List


def extract_table(element):
    # return f"{element['data_source']}.{element['table_name']}"
    return element['arguments']['table_name']


def gather_join_restiction(element):
    table0, column0 = element['arguments']['attribute_name0'].split('.')
    table1, column1 = element['arguments']['attribute_name1'].split('.')

    return SimpleNamespace(table0=table0, column0=column0, table1=table1, column1=column1)


def build_join(join_restriction: SimpleNamespace, tables_used: List[str]):
    prefix = ""
    if len(tables_used) == 0:
        prefix = join_restriction.table0
        tables_used.append(prefix)

    table_to_use = join_restriction.table1

    if table_to_use in tables_used:
        table_to_use = join_restriction.table0

    tables_used.append(table_to_use)

    return f"{prefix} join {table_to_use} on {join_restriction.table0}.{join_restriction.column0} = {join_restriction.table1}.{join_restriction.column1}"


def assemble_joins(join_restrictions):
    joins = []
    tables_used = []

    # the first case is a special case: here we add both tables to the join
    joins.append(build_join(join_restrictions.pop(0), tables_used))

    # we need to make sure the joins are in the right order, therefore we try as long
    # as necessary. with the pop(0) and the append() we basically implement a FIFO Queue
    while len(join_restrictions) > 0:
        join_restriction = join_restrictions.pop(0)

        # If neither of the two tables are already used, we need need to try another join_restriction first
        # This is because only one table gets added per join.
        if join_restriction.table0 in tables_used or join_restriction.table1 in tables_used:
            joins.append(build_join(join_restriction, tables_used))
        else:
            join_restrictions.append(join_restriction)

    return joins


def build_filter(element, is_first):
    prefix = 'where' if is_first else 'and'
    return f"{prefix} {element['arguments']['attribute_name']} {element['arguments']['operation']} '{element['arguments']['value']}'"


def build_projection(element, distinct):
    projection_functions = {
        'Average': 'avg',
        'Count': 'count',
        'Sum': 'sum'
    }

    if len(element['arguments']) > 0:
        column = element['arguments']['attribute_name']
    elif distinct:
        column = distinct
    else:
        raise ValueError('Neither distinct nor a column to select. Is this correct?')

    return f"{projection_functions[element['operation']]}({column})"


def extract_distinct(element):
    return f"distinct {element['arguments']['attribute_name']}"


def tree_to_sql(tree: dict):
    join_restrictions = []
    filters = []
    projections = []
    distinct = ''
    is_empty = False

    for element in tree:

        operation = element['operation']

        if operation == 'Merge':
            join_restrictions.append(gather_join_restiction(element))
        elif operation == 'Filter':
            filters.append(build_filter(element, len(filters) == 0))
        elif operation == 'Average':
            projections.append(build_projection(element, distinct))
        elif operation == 'Count':
            projections.append(build_projection(element, distinct))
        elif operation == 'Sum':
            projections.append(build_projection(element, distinct))
        elif operation == 'Distinct':
            distinct = extract_distinct(element)
        elif operation == 'IsEmpty':
            is_empty = True
        elif operation == 'GetData':
            # no need to do anything here, we get all information for a join from the Merge operation
            pass
        elif operation == 'ExtractValues':
            # Not sure that's correct - is it possible that this node is at the top of a tree?
            pass
        elif operation == 'Done':
            # TODO That might be a mistake as there might be queries without any aggregation function AND without distinct. In that case the Done might signal that the last ExtractValues is the projection.
            pass
        else:
            raise ValueError(f'unknown operation "{operation}"')

    if len(join_restrictions) > 0:
        joins = assemble_joins(join_restrictions)
    else:
        joins = []

    projection_string = ", ".join(projections) if len(projections) > 0 else distinct

    join_string = "\n".join(joins)
    filter_string = "\n".join(filters)

    assembled_query = f"select {projection_string}\n" \
                 f"from {join_string}\n" \
                 f"{filter_string}"

    if is_empty:
        assembled_query = f"select not exists({assembled_query})"

    return assembled_query


def main():
    with open('data/cordis/trees/5fb23d0b913f25fdb2030fcc.json', 'r', encoding='utf8') as f:
        tree = json.load(f)

    full_query = tree_to_sql(tree)

    print(full_query)


if __name__ == '__main__':
    main()
