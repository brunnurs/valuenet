def infer_from_clause(table_names, graph, columns):
    tables = list(table_names.keys())

    if len(tables) == 1:  # no JOINS needed - just return the simple "FROM" clause.
        return "FROM {} AS {}".format(tables[0], table_names[tables[0]])
    else:  # we have to deal with multiple tables - and find the shortest path between them
        join_clauses, cross_join_clauses = generate_path_by_graph(graph, table_names, tables)

    if len(_tables_in_join_clauses(join_clauses)) >= 3:
        join_clauses = _find_and_remove_star_table(columns, join_clauses)

    stringified_join_clauses = []

    for idx, (start, start_alias, end, end_alias, entry_column, exit_column) in enumerate(join_clauses):

        # the first case is kind of an exception case, as we need to write two tables, for example: "A AS T1 JOIN B AS T2 ON ....".
        # All the following joins will only be "... JOIN T2 ON ...."
        if idx == 0:
            stringified_join_clauses.append("{} AS {} JOIN {} AS {} ON {}.{} = {}.{}".format(start,
                                                                                           start_alias,
                                                                                           end,
                                                                                           end_alias,
                                                                                           start_alias,
                                                                                           entry_column,
                                                                                           end_alias,
                                                                                           exit_column))
        else:
            stringified_join_clauses.append("JOIN {} AS {} ON {}.{} = {}.{}".format(end,
                                                                                  end_alias,
                                                                                  start_alias,
                                                                                  entry_column,
                                                                                  end_alias,
                                                                                  exit_column))

    # that's the cross-join exception cases. We have to add them for syntactical correctness, even though it will not result
    # in a good query at execution.
    for table, table_alias in cross_join_clauses:
        if len(stringified_join_clauses) == 0:
            stringified_join_clauses.append("{} AS {}".format(table, table_alias))
        else:
            stringified_join_clauses.append("JOIN {} AS {}".format(table, table_alias))

    return 'FROM ' + ' '.join(stringified_join_clauses)


def generate_path_by_graph(graph, table_names, tables):
    join_clause = list()

    cross_joins, tables_handled_by_cross_joins = _handle_standalone_tables(graph, table_names, tables)

    tables_cleaned = [table for table in tables if table not in tables_handled_by_cross_joins]

    idx = 0
    edges = []
    # We always deal with two tables at the time and try to find the shortest path between them. This might be over-simplified
    # as there could be a more optimal path between all tables (see Steiner Graph), but practically it doesn't matter so much.
    while idx < len(tables_cleaned) - 1:
        start_table = tables_cleaned[idx]
        end_table = tables_cleaned[idx + 1]

        edges_for_this_path = graph.dijkstra(start_table, end_table)
        if edges_for_this_path:
            edges.extend(edges_for_this_path)
        else:
            raise Exception(f"We could not find a path between table '${start_table}' and '${end_table}'. This query can"
                            f"not work. Make sure you allow only questions in a fully connected schema!")

        idx += 1

    # now there might be duplicates - as parts of the path from A to C might be the same as from A to B.
    # be aware that, as we only consider INNER JOINS, A <-> B is equal to B <-> A! So we also have to remove this edges.
    edges_deduplicated = _deduplicate_edges(edges)

    # "max_alias_idx" is the highest alias in the current join_clause (e.g. "T1, T2, T3"). We need to find it as we
    # wanna add new tables and continue the enumeration properly.
    max_alias_idx = _get_max_alias(table_names)

    # now for each edge we now have to add both, the start table and the end table to the join_clause (including the PK/FK-columns).
    for edge in edges_deduplicated:
        if edge.start not in table_names:
            table_names[edge.start] = 'T' + str(max_alias_idx + 1)
            max_alias_idx += 1

        if edge.end not in table_names:
            table_names[edge.end] = 'T' + str(max_alias_idx + 1)
            max_alias_idx += 1

        join_clause.append((edge.start,
                            table_names[edge.start],
                            edge.end,
                            table_names[edge.end],
                            edge.entry_column,
                            edge.exit_column))

    return join_clause, cross_joins


def _handle_standalone_tables(graph, table_names, tables):
    join_clause = []
    tables_handled = []

    # there is a few rare cases of tables without connections to others - which will then obviously not be part of the graph.
    # as we can't properly handle this cases, we just have to do a stupid cross-join with them
    for table in tables:
        if table not in graph.vertices:
            join_clause.append((table, table_names[table]))
            tables_handled.append(table)

    remaining_tables = [t for t in table_names if t not in tables_handled]

    # if there is only one table left after removing all the others, we can't use a graph anymore - so we need to do use a cross join as well.
    if len(remaining_tables) == 1:
        join_clause.append((remaining_tables[0], table_names[remaining_tables[0]]))
        tables_handled.append(remaining_tables[0])

    return join_clause, tables_handled


def _get_max_alias(table_names):
    max_key = 1
    for t, k in table_names.items():
        _k = int(k[1:])
        if _k > max_key:
            max_key = _k
    return max_key + 10


def _find_and_remove_star_table(columns, join_clause):
    """
    Starting from 3 tables we have to deal with the "star-table" effect - a join with a joining table where we only wanna know e.g. the count(*) of the third table.
    In that case we don't need to join the third table - we just do a count over the join with the joining table.
    In general, the additional join is not an issue - but is seen as incorrect by the spider-evaluation and therefore we have to remove it.

    Example:
        SELECT T2.concert_name ,  T2.theme ,  count(*) FROM singer_in_concert AS T1 JOIN concert AS T2 ON T1.concert_id  =  T2.concert_id GROUP BY T2.concert_id ---> GOOD
        SELECT T1.concert_Name, T1.Theme, count(*) FROM concert AS T1 JOIN singer_in_concert AS T3 JOIN singer AS T2  GROUP BY T1.concert_ID  -----> BAD, REMOVE "singer" join.
    """

    # unfortunately auto tuple unpacking doesn't work anymore in python 3, therefore this comment: a "column" contains the 3 elements "aggregator, "column name", "table".
    star_tables = list(map(lambda column: column[2], filter(lambda column: column[1] == '*', columns)))

    # remove duplicates
    star_tables = list(set(star_tables))

    assert len(star_tables) <= 1, "The case of having multiple star-joins is currently not supported (and not part of the spider-dataset)"

    if len(star_tables) == 1:
        star_table = star_tables[0]

        # we need to make sure the table we try to remove is not used at any other place - e.g. in the SELECT or in the WHERE clause.
        # only then we can safely remove it
        if len(list(filter(lambda column: column[1] != '*' and column[2] == star_table, columns))) == 0:

            # we only remove star-tables if they are the start or end table in the graph.
            # remember, an join_clause tuple looks like this: (start, start_alias, end, end_alias, entry_column, exit_column)
            start_edge = join_clause[0]
            start_edge_from, _, start_edge_to, _, _, _ = start_edge

            end_edge = join_clause[len(join_clause) - 1]
            end_edge_from, _, end_edge_to, _, _, _ = end_edge

            if start_edge_from == star_table:
                if second_table_in_edge_is_availabe_elswhere(start_edge_to, join_clause[1:]):
                    return join_clause[1:]

            if end_edge_to == star_table:
                if second_table_in_edge_is_availabe_elswhere(end_edge_from, join_clause[:-1]):
                    return join_clause[:-1]
    return join_clause


def second_table_in_edge_is_availabe_elswhere(second_table, remaining_edges):
    """
    By removing an edge, we basically remove two tables. If there schema is a "normal" schema, where the edges are "A --> B", "B --> C"
    this is not an issue.

    We we though have a non-linear schema, like "A --> B", "A --> C" we can't just remove the first edge - we would loose B completely!
    To avoid this we make sure the second table in the edge we plan to remove is available in another edge.

    A schema where we have to deal with this issue is e.g. "flight_2", where two relations go from "flights" to "airports".
    """
    for edge in remaining_edges:
        start, _, end, _, _, _ = edge
        if second_table == start or second_table == end:
            return True

    return False


def _deduplicate_edges(edges):
    deduplicated = []

    for e1 in edges:
        found_match = False
        for e2 in deduplicated:
            # make sure two edges do not match - while paying no attention to the direction of the edge!
            # more complex might make it necessary to also include the foreign key/primary key here, as you could theoretically have multiple relationships between two tables.
            if (e1.start == e2.start and e1.end == e2.end) or (e1.start == e2.end and e1.end == e2.start):
                found_match = True

        if not found_match:
            deduplicated.append(e1)

    return deduplicated


def _tables_in_join_clauses(join_clauses):
    unique_tables = set()
    for clause in join_clauses:
        start_table, _, end_table, _, _, _ = clause
        unique_tables.add(start_table)
        unique_tables.add(end_table)

    return list(unique_tables)
