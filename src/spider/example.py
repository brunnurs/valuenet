import copy

from intermediate_representation.semQL import C, T, A, V
import neural_network_utils as nn_utils


class Example:
    def __init__(self, src_sent, tgt_actions=None, vis_seq=None, tab_cols=None, col_num=None, sql=None,
                 one_hot_type=None, col_hot_type=None, schema_len=None, tab_ids=None,
                 table_names=None, table_len=None, col_table_dict=None, cols=None,
                 table_col_name=None, table_col_len=None,
                 col_pred=None, tokenized_src_sent=None,
                 values=None
                 ):
        """

        @param src_sent: [['what'], ['are'], ['column', 'name'], ['of'], ['state'], ['where'], ['at'], ['least'], ['value', '3'], ['table', 'head'], ['were'], ['born'], ['?']]
        @param tgt_actions: [Root1(3), Root(3), Sel(0), N(0), A(none), C(8), T(1), Filter(Filter >= A), A(count), C(0), T(1)]
        @param vis_seq: a tripe of the unprocessed question, a list of columns and the raw query. Does not seem to be used anywhere.
        @param tab_cols: [['count', 'number', 'many'], ['department', 'id'], ['name'], ['creation'], ['ranking'], ['budget', 'in', 'billion'], ['num', 'employee'], ['head', 'id'], ['born', 'state'], ['age'], ['temporary', 'acting']]
        @param col_num:  11
        @param sql: 'SELECT born_state FROM head GROUP BY born_state HAVING count(*)  >=  3'
        @param one_hot_type: The one_hot_type has the same length as src_sent (tokenized question after standard NLP-pre-processing). It contains a one-hot-encoded vector of length 6 to represent information like 0:"table", 1:"column", 2:"agg" 3:"MORE", 4:"MOST", 5:"value". Tokens with NONE will not be represented in this array.
        @param col_hot_type: Has the same length as tab_cols (columns) and is indicating the type of this column, meaning if it is a partial match (#0) or an exact match (#1) of a column. An exact match (a 5 in #1) often also has some values in #1, so it's not one hot encoding. Not sure what #2 and #3 is used for, as it's never used with all spider data. This data will later be used for schema encoding, as the 3rd part (the "phi") in the paper
        @param schema_len: ---- not used in constructor ----
        @param tab_ids: ---- not used in constructor ----
        @param table_names: [['department'], ['head'], ['management']]. Multi-word tables would be split.
        @param table_len: 3
        @param col_table_dict: this dict is telling for each column in what table it appears. So the key is the idx of the column, the values the idx of the tables.
        @param cols: ['*', 'department id', 'name', 'creation', 'ranking', 'budget in billions', 'num employees', 'head id', 'name', 'born state', 'age', 'department id', 'head id', 'temporary acting']
        @param table_col_name: [['department', 'id', 'name', 'creation', 'ranking', 'budget', 'in', 'billion', 'num', 'employee'], ['head', 'id', 'name', 'born', 'state', 'age'], ['department', 'id', 'head', 'id', 'temporary', 'acting']]
        @param table_col_len: 3
        @param col_pred: ---- not used in constructor ----
        @param tokenized_src_sent: # no idea why we use this here again... its "col_set_type" from above.
        @param values: The query-values used in this example. Can be a string (e.g."USA"), a numerical value (e.g. 1.2), a data ('31-03-2019') or even more exotic formats.
        """
        self.src_sent = src_sent
        self.tokenized_src_sent = tokenized_src_sent
        self.vis_seq = vis_seq
        self.tab_cols = tab_cols
        self.col_num = col_num
        self.sql = sql
        self.one_hot_type = one_hot_type
        self.col_hot_type = col_hot_type
        self.schema_len = schema_len
        self.tab_ids = tab_ids
        self.table_names = table_names
        self.table_len = table_len
        self.col_table_dict = col_table_dict
        self.cols = cols
        self.table_col_name = table_col_name
        self.table_col_len = table_col_len
        self.col_pred = col_pred
        self.values = values
        self.tgt_actions = tgt_actions
        self.truth_actions = copy.deepcopy(tgt_actions)

        self.sketch = list()
        if self.truth_actions:
            for ta in self.truth_actions:
                if isinstance(ta, C) or isinstance(ta, T) or isinstance(ta, A) or isinstance(ta, V):
                    continue
                self.sketch.append(ta)


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

        if examples[0].tgt_actions:
            self.max_action_num = max(len(e.tgt_actions) for e in self.examples)
            self.max_sketch_num = max(len(e.sketch) for e in self.examples)

        self.src_sents = [e.src_sent for e in self.examples]
        self.src_sents_len = [len(e.src_sent) for e in self.examples]
        self.tokenized_src_sents = [e.tokenized_src_sent for e in self.examples]
        self.tokenized_src_sents_len = [len(e.tokenized_src_sent) for e in examples]
        self.src_sents_word = [e.src_sent for e in self.examples]
        self.table_sents_word = [[" ".join(x) for x in e.tab_cols] for e in self.examples]

        self.schema_sents_word = [[" ".join(x) for x in e.table_names] for e in self.examples]

        # TODO seems that the "src_type" (or "one_hot_type" as it is called before) is not used at all. This makes sense as it is anway part of "src_sents" (the "table"/"column"/etc.) prefix. Remove it if time.
        self.src_type = [e.one_hot_type for e in self.examples]
        self.col_hot_type = [e.col_hot_type for e in self.examples]
        self.table_sents = [e.tab_cols for e in self.examples]
        self.col_num = [e.col_num for e in self.examples]
        self.tab_ids = [e.tab_ids for e in self.examples]
        self.table_names = [e.table_names for e in self.examples]
        self.table_len = [e.table_len for e in examples]
        self.col_table_dict = [e.col_table_dict for e in examples]
        self.table_col_name = [e.table_col_name for e in examples]
        self.table_col_len = [e.table_col_len for e in examples]
        self.col_pred = [e.col_pred for e in examples]
        self.values = [e.values for e in examples]
        self.n_values = [len(e.values) for e in examples]

        self.grammar = grammar
        self.cuda = cuda

    def __len__(self):
        return len(self.examples)

    def table_dict_mask(self, table_dict):
        return nn_utils.table_dict_to_mask_tensor(self.table_len, table_dict, cuda=self.cuda)

    @cached_property
    def pred_col_mask(self):
        return nn_utils.pred_col_mask(self.col_pred, self.col_num)

    @cached_property
    def schema_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.table_len, cuda=self.cuda)

    @cached_property
    def table_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def value_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.n_values, cuda=self.cuda)

    @cached_property
    def table_appear_mask(self):
        return nn_utils.appear_to_mask_tensor(self.col_num, cuda=self.cuda)

    @cached_property
    def table_unk_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.col_num, cuda=self.cuda, value=None)

    @cached_property
    def src_token_mask(self):
        return nn_utils.length_array_to_mask_tensor(self.src_sents_len, cuda=self.cuda)
