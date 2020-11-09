import copy

from intermediate_representation.semQL import C, T, A, V
import neural_network_utils as nn_utils


class Example:
    def __init__(self, question_tokens, semql_actions=None, column_tokens=None, n_columns=None, sql=None, column_matches=None,
                 tables=None, n_tables=None, column_table_dict=None, columns=None, columns_per_table=None, values=None):
        """

        @param question_tokens: [['what'], ['are'], ['column', 'name'], ['of'], ['state'], ['where'], ['at'], ['least'], ['value', '3'], ['table', 'head'], ['were'], ['born'], ['?']]
        @param semql_actions: [Root1(3), Root(3), Sel(0), N(0), A(none), C(8), T(1), Filter(Filter >= A), A(count), C(0), T(1)]
        @param column_tokens: [['count', 'number', 'many'], ['department', 'id'], ['name'], ['creation'], ['ranking'], ['budget', 'in', 'billion'], ['num', 'employee'], ['head', 'id'], ['born', 'state'], ['age'], ['temporary', 'acting']]
        @param n_columns:  11
        @param sql: 'SELECT born_state FROM head GROUP BY born_state HAVING count(*)  >=  3'
        @param column_matches: Has the same length as tab_cols (columns) and is indicating the colum matches --> how many times a column has ben "hit" when comparing with the question. This data will later be used for schema encoding, as the 3rd part (the "phi") in the paper
        @param tables: [['department'], ['head'], ['management']]. Multi-word tables would be split.
        @param n_tables: 3
        @param column_table_dict: this dict is telling for each column in what table it appears. So the key is the idx of the column, the values the idx of the tables.
        @param columns: ['*', 'department id', 'name', 'creation', 'ranking', 'budget in billions', 'num employees', 'head id', 'name', 'born state', 'age', 'department id', 'head id', 'temporary acting']
        @param columns_per_table: [['department', 'id', 'name', 'creation', 'ranking', 'budget', 'in', 'billion', 'num', 'employee'], ['head', 'id', 'name', 'born', 'state', 'age'], ['department', 'id', 'head', 'id', 'temporary', 'acting']]
        @param values: The query-values used in this example. Can be a string (e.g."USA"), a numerical value (e.g. 1.2), a data ('31-03-2019') or even more exotic formats.
        """
        self.question_tokens = question_tokens
        self.column_tokens = column_tokens
        self.n_columns = n_columns
        self.sql = sql
        self.column_matches = column_matches
        self.tables = tables
        self.n_tables = n_tables
        self.column_table_dict = column_table_dict
        self.columns = columns
        self.columns_per_table = columns_per_table
        self.semql_actions = semql_actions
        self.values = values

        self.sketch = list()
        if self.semql_actions:
            for action in self.semql_actions:
                if isinstance(action, C) or isinstance(action, T) or isinstance(action, A) or isinstance(action, V):
                    continue
                self.sketch.append(action)


class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


class Batch(object):
    def __init__(self, examples, grammar, cuda=False):
        self.examples = examples

        if examples[0].semql_actions:
            self.max_action_num = max(len(e.semql_actions) for e in self.examples)
            self.max_sketch_num = max(len(e.sketch) for e in self.examples)

        self.all_question_tokens = [e.question_tokens for e in self.examples]
        # the +1 represents the extra separator token after the end of the question. Not sure yet it is really necessary.
        self.all_question_tokens_len = [len(e.question_tokens) + 1 for e in self.examples]

        self.all_column_matches = [e.column_matches for e in self.examples]
        self.all_column_tokens = [e.column_tokens for e in self.examples]
        self.all_n_columns = [e.n_columns for e in self.examples]
        self.all_table_names = [e.tables for e in self.examples]
        self.all_n_tables = [e.n_tables for e in examples]
        self.all_column_table_dict = [e.column_table_dict for e in examples]
        self.all_columns_per_table = [e.columns_per_table for e in examples]
        self.values = [e.values for e in examples]
        self.n_values = [len(e.values) for e in examples]

        self.grammar = grammar
        self.cuda = cuda

    def __len__(self):
        return len(self.examples)

    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.all_n_tables, table_dict, cuda=self.cuda)

    @cached_property
    def schema_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.all_n_tables, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.all_n_columns, cuda=self.cuda)

    @cached_property
    def value_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.n_values, cuda=self.cuda)

    @cached_property
    def table_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.all_n_columns)

    @cached_property
    def table_unk_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.all_n_columns, cuda=self.cuda, value=None)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.all_question_tokens_len, cuda=self.cuda)
