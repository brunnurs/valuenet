def infer_from_clause(table_names, schema, columns):
    tables = list(table_names.keys())
    # print(table_names)
    start_table = None
    end_table = None
    join_clause = list()
    if len(tables) == 1:
        join_clause.append((tables[0], table_names[tables[0]]))
    elif len(tables) == 2:
        use_graph = True
        # print(schema['graph'].vertices)
        for t in tables:
            if t not in schema['graph'].vertices:
                use_graph = False
                break
        if use_graph:
            start_table = tables[0]
            end_table = tables[1]
            tables_with_in_between_tables = list(schema['graph'].dijkstra(tables[0], tables[1]))

            # "max_key" is the highest alias in the current join_clause (e.g. "T1, T2, T3"). We need to find it as we
            # wanna add new tables and continue the enumeration properly.
            max_key = get_max_key(table_names)

            add_in_between_tables_to_join_clause(join_clause, max_key, table_names, tables_with_in_between_tables)
        else:
            # TODO: there is 33 cases on the dev-set which can somehow not be handled with the Graph. Why is this?
            print("we don't use the graph. Why?")
            join_clause = list()
            for t in tables:
                join_clause.append((t, table_names[t],))
    else:
        #TODO: handle this case with the Graph, same as the 2-table case. 15 cases on the dev-set!
        for t in tables:
            join_clause.append((t, table_names[t],))

        # print('More than 2 table! db: {}            tables: {}'.format(schema['db_id'], join_clause))

    if len(join_clause) >= 3:
        star_table = None
        for agg, col, tab in columns:
            if col == '*':
                star_table = tab
                break
        if star_table is not None:
            star_table_count = 0
            for agg, col, tab in columns:
                if tab == star_table and col != '*':
                    star_table_count += 1
            if star_table_count == 0 and ((end_table is None or end_table == star_table) or (start_table is None or start_table == star_table)):
                # Remove the table the rest tables still can join without star_table
                new_join_clause = list()
                for t in join_clause:
                    if t[0] != star_table:
                        new_join_clause.append(t)
                join_clause = new_join_clause

    join_clause = ' JOIN '.join(['%s AS %s' % (jc[0], jc[1]) for jc in join_clause])
    return 'FROM ' + join_clause


def add_in_between_tables_to_join_clause(join_clause, max_key, table_names, tables_with_in_between_tables):
    for t in tables_with_in_between_tables:
        if t not in table_names:
            table_names[t] = 'T' + str(max_key + 1)
            max_key += 1
        join_clause.append((t, table_names[t],))


def get_max_key(table_names):
    max_key = 1
    for t, k in table_names.items():
        _k = int(k[1:])
        if _k > max_key:
            max_key = _k
    return max_key
