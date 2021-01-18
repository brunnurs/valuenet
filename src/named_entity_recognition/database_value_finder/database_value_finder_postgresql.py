import psycopg2

from named_entity_recognition.database_value_finder.database_value_finder import DatabaseValueFinder


class DatabaseValueFinderPostgreSQL(DatabaseValueFinder):
    def __init__(self, database_name, db_schema_information, connection_config, max_results=10):

        super().__init__(database_name, db_schema_information, max_results)

        self.db_host = connection_config['database_host']
        self.db_port = connection_config['database_port']
        self.db_user = connection_config['database_user']
        self.db_password = connection_config['database_password']
        self.db_options = f"-c search_path={connection_config['database_schema']},public"

        # as this thresholds are highly depending on the database specific implementation, it needs to be provided here
        self.exact_match_threshold = 1.0  # be a ware that an exact match is not case sensitive
        self.high_similarity_threshold = 0.75
        self.medium_similarity_threshold = 0.7

    def find_similar_values_in_database(self, potential_values, include_primary_keys):
        matching_values = set()

        # right now the similarity search on PostgreSQL supports only text columns due to the special indices
        table_text_column_mapping = self._get_relevant_columns(include_primary_keys, text_columns_only=True)

        conn = psycopg2.connect(database=self.database, user=self.db_user, password=self.db_password, host=self.db_host, port=self.db_port, options=self.db_options)

        for table, columns in table_text_column_mapping.items():
            for column in columns:
                matches = self._find_matches_in_column(table, column, potential_values, conn)
                matching_values.update(matches)

        conn.close()

        return self._top_n_results(matching_values)

    def _find_matches_in_column(self, table, column, potential_values, connection):
        query = self._assemble_query(column, table, potential_values)
        # print(query)
        # toc = TicToc()
        # toc.tic()

        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        # toc.toc(f"{table}.{column} took")
        # print(len(rows))

        # r[0] is always the value of the column we query. r[1:] represents the similarity with all potential values.
        # Keep in mind that we only get the result back, if one of the similarities was higher than the similiarity-threshold
        # for this potential value. By using the max() function we therefore always get the similarity of the potential value
        # that matched best to the value of the column
        return list(map(lambda r: (r[0], max(r[1:]), column, table), rows))

    @staticmethod
    def _assemble_query(column, table, potential_values):
        """
        What we want to return in this method is a nested query looking like the following:
        select title, sim_v1, sim_v2, sim_v3 from
            (SELECT DISTINCT title,
                            similarity(title, 'Nural Language Energy for Promoting CONSUMER Sustainable Behaviour') as sim_v1,
                            similarity(title, 'dummy1')                                                             as sim_v2,
                            similarity(title, 'dummy2')                                                             as sim_v3
            FROM unics_cordis.projects
            WHERE title % 'Nural Language Energy for Promoting CONSUMER Sustainable Behaviour'
               OR title % 'dummy1'
               OR title % 'dummy2') as sub_query
        where sim_v1 >= 0.9 OR sim_v2 >= 0.5 OR sim_v2 >= 0.54

        Why is the nested query necessary? The special "gist_trgm_ops"-index we create for all text columns in the database works only with the % operator, not by using a WHERE similiarity(a,b) > x restriction. We therefore
        need the inner query to massively reduce the result set before applying the exact similarity restrictions in order to make this query fast.
        Be aware that the % operator works with the internal threshold set by set_limit(y) and returned by show_limit(). We therefore need to set the lowest possible threshold here (e.g. 0.499) and then use the other thresholds
        to further restrict the result set in the outer query.
        @return:
        """
        outer_query_selection = ', '.join([f'sim_v{i}' for i in range(0, len(potential_values))])
        outer_query_filter = ' OR '.join([f"sim_v{idx} >= {value[1]}" for idx, value in enumerate(potential_values)])

        inner_query_selection = ', '.join([f"similarity({column},'{value[0]}') as sim_v{idx}" for idx, value in enumerate(potential_values)])
        inner_query_filter = ' OR '.join([f"{column} % '{value[0]}'" for value in potential_values])

        # similarity_listening = reduce(lambda current, next_value: current + f" OR similarity({column},'{next_value[0]}') >= {next_value[1]}", potential_values, '')

        full_query = f"SELECT {column}, {outer_query_selection} FROM " \
                     f"(SELECT DISTINCT {column}, " \
                     f"{inner_query_selection} " \
                     f"FROM {table} " \
                     f"WHERE {inner_query_filter}) AS sub_query " \
                     f"WHERE {outer_query_filter}"

        # print(full_query)

        return full_query
