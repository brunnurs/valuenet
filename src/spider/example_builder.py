import copy

import numpy
import numpy as np
from nltk import WordNetLemmatizer

from intermediate_representation import lf
from preprocessing.pre_process import lemmatize_list
from spider.example import Example

# Take care, this imports are necessary due to the dynamic "eval()" command further down
from intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1

wordnet_lemmatizer = WordNetLemmatizer()


def build_column_matches_array(column_matches):
    """
    Here we map column matches (so the number of times a column got "hit" in the question) a simple 4d array.
    There might be smarter approaches to think about (e.g. embeddings?)
    """
    column_matches_array = np.zeros((len(column_matches), 4))

    for idx, match in enumerate(column_matches):
        column_matches_array[idx][0] = match['partial_column_match']
        column_matches_array[idx][1] = 5 if match['full_column_match'] else 0
        column_matches_array[idx][2] = 5 if match['full_value_match'] else 0
        # this actually happens never, we might remove it!
        column_matches_array[idx][3] = match['partial_value_match']

    return column_matches_array


def build_example(sql, table_data):
    # The table data contains detailed information about that database. This includes tables, table headers (columns)
    # and also foreign/primary keys.
    table = table_data[sql['db_id']]

    column_matches = build_column_matches_array(sql['column_matches'])

    _, table_names = lemmatize_list(table['table_names'])

    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]

    _, col_set_iter = lemmatize_list(sql['col_set'])
    _, col_iter = lemmatize_list(tab_cols)

    # this dict is telling for each column in what table it appears. So the key is the idx of the column, the values the idx of the tables.
    # example: the key 0 will appear in all tables (e.g. [1,2,3,4,5]) as it's the special column "*".
    # Most others will only appear one, but the id's (which are used as primary key / foreign key) will appear also multiple times.
    col_table_dict = _get_col_table_dict(tab_cols, tab_ids, sql)
    # a simple list with sublists for each table, containing all the columns in that table.
    table_col_name = _get_table_colNames(tab_ids, col_iter)

    # this field contains the special column "*", referring to all columns in all tables. Not sure yet why we replace it with this special content.
    col_set_iter[0] = ['count', 'number', 'many']

    # in the pre-processing (see sql2SemQL.py) we parse the sql for each example to the SemQL-AST language. We then serialize it to a string.
    # Here we do the opposite: we deserialize the SemQL-Query by dynamically create the right objects based on the string.
    rule_label = None
    if 'rule_label' in sql:
        # Example: eval("Root1(3)") will dynamically create an instance of class Root1 with the constructor argument 3.
        rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]

        if is_valid(rule_label, col_table_dict=col_table_dict, sql=sql) is False:
            raise RuntimeError("Invalid rule_label: {}. We don't use this sample".format(sql['rule_label']))

    example = Example(
        src_sent=[[token] for token in sql['question_toks']],
        col_num=len(col_set_iter),
        tab_cols=col_set_iter,
        sql=sql['query'],
        col_hot_type=column_matches,
        table_names=table_names,
        table_len=len(table_names),
        col_table_dict=col_table_dict,
        cols=tab_cols,
        table_col_name=table_col_name,
        table_col_len=len(table_col_name),
        tgt_actions=rule_label
    )
    example.sql_json = copy.deepcopy(sql)

    return example


def _get_col_table_dict(tab_cols, tab_ids, sql):
    table_dict = {}
    for c_id, c_v in enumerate(sql['col_set']):
        for cor_id, cor_val in enumerate(tab_cols):
            if c_v == cor_val:
                table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]

    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]
    return col_table_dict


def _get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result


def is_valid(rule_label, col_table_dict, sql):
    try:
        lf.build_tree(copy.copy(rule_label))  # here we rebuild a tree from the ist with rules
    except:
        print(rule_label)

    flag = False
    for r_id, rule in enumerate(rule_label):
        if type(rule) == C:
            try:
                assert rule_label[r_id + 1].id_c in col_table_dict[rule.id_c], print(sql['question'])
            except:
                flag = True
                print(sql['question'])
    return flag is False
