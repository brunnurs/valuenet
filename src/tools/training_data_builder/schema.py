class SchemaIndex:
    """
    Simple schema which maps table&column to the column index. This format is consistent with the old spider data.
    Why can't we use the "Schema" class from process_sql.py? This class uses a new format which does not work with the
    column indices anymore. While this is in general OK, it is inconsistent with all scripts in this projects processing
    the sql-json struct.
    """
    def __init__(self, schema_map, column_names_original, table_names_original):
        self._schema = schema_map
        self._idMap = self._map(self._schema, column_names_original, table_names_original)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, column_names_original, table_names_original):
        id_map = {'*': 0}

        for table, columns in schema.items():
            for column in columns:
                id_map[table.lower() + "." + column.lower()] = self._lookup_idx(table, column, column_names_original, table_names_original)

        for table in schema:
            id_map[table.lower()] = table_names_original.index(table)

        return id_map

    @staticmethod
    def _lookup_idx(table, column, column_names_original, table_names_original):
        """
        Lookup a column/table combination in the schema and return the column index
        """
        for idx, (table_idx, column_name) in enumerate(column_names_original):
            if column_name == column:
                if table_names_original[table_idx] == table:
                    return idx

        raise ValueError(f'We could not find index for column "{column}", table "{table}"')


def build_schema_mapping(schema: dict) -> dict:
    table_schema_mapping = {}

    # first column is always "*" which we dont need
    for table_idx, column_name in schema['column_names_original'][1:]:
        table_name = schema['table_names_original'][table_idx]

        if table_name in table_schema_mapping:
            table_schema_mapping[table_name].append(column_name)
        else:
            table_schema_mapping[table_name] = [column_name]

    return table_schema_mapping
