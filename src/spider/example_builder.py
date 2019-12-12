import copy
import numpy as np
from nltk import WordNetLemmatizer

from src.intermediate_representation import lf
from src.spider.example import Example

# Take care, this imports are necessary due to the dynamic "eval()" command further down
from src.intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1

wordnet_lemmatizer = WordNetLemmatizer()


def build_example(sql, table_data):
    # The table data contains detailed information about that database. This includes tables, table headers (columns)
    # and also foreign/primary keys.
    table = table_data[sql['db_id']]

    process_dict = _process(sql, table)

    for c_id, col_ in enumerate(process_dict['col_set_iter']):
        for q_id, ori in enumerate(process_dict['q_iter_small']):
            if ori in col_:
                process_dict['col_set_type'][c_id][0] += 1

    _schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                    process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'], sql)

    col_table_dict = _get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], sql)
    table_col_name = _get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])

    process_dict['col_set_iter'][0] = ['count', 'number', 'many']

    rule_label = None
    if 'rule_label' in sql:
        # eval("Root1(3)") will dynamically create an instance of class Root1 with the constructor argument 3.
        rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]

        if is_valid(rule_label, col_table_dict=col_table_dict, sql=sql) is False:
            raise Exception("Invalid rule_label! Deal with it.")

    example = Example(
        src_sent=process_dict['question_arg'],
        col_num=len(process_dict['col_set_iter']),
        vis_seq=(sql['question'], process_dict['col_set_iter'], sql['query']),
        tab_cols=process_dict['col_set_iter'],
        sql=sql['query'],
        one_hot_type=process_dict['one_hot_type'],
        col_hot_type=process_dict['col_set_type'],
        table_names=process_dict['table_names'],
        table_len=len(process_dict['table_names']),
        col_table_dict=col_table_dict,
        cols=process_dict['tab_cols'],
        table_col_name=table_col_name,
        table_col_len=len(table_col_name),
        tokenized_src_sent=process_dict['col_set_type'],
        tgt_actions=rule_label
    )
    example.sql_json = copy.deepcopy(sql)

    return example


def _process(sql, table):
    process_dict = {}

    origin_sql = sql['question_toks']
    table_names = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in table['table_names']]

    sql['pre_sql'] = copy.deepcopy(sql)

    tab_cols = [col[1] for col in table['column_names']]
    tab_ids = [col[0] for col in table['column_names']]

    col_set_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(' ')] for x in sql['col_set']]
    col_iter = [[wordnet_lemmatizer.lemmatize(v).lower() for v in x.split(" ")] for x in tab_cols]
    q_iter_small = [wordnet_lemmatizer.lemmatize(x).lower() for x in origin_sql]
    question_arg = copy.deepcopy(sql['question_arg'])
    question_arg_type = sql['question_arg_type']
    one_hot_type = np.zeros((len(question_arg_type), 6))

    col_set_type = np.zeros((len(col_set_iter), 4))

    process_dict['col_set_iter'] = col_set_iter
    process_dict['q_iter_small'] = q_iter_small
    process_dict['col_set_type'] = col_set_type
    process_dict['question_arg'] = question_arg
    process_dict['question_arg_type'] = question_arg_type
    process_dict['one_hot_type'] = one_hot_type
    process_dict['tab_cols'] = tab_cols
    process_dict['tab_ids'] = tab_ids
    process_dict['col_iter'] = col_iter
    process_dict['table_names'] = table_names

    return process_dict


def _schema_linking(question_arg, question_arg_type, one_hot_type, col_set_type, col_set_iter, sql):
    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            question_arg[count_q] = ['table'] + question_arg[count_q]
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            try:
                col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                question_arg[count_q] = ['column'] + question_arg[count_q]
            except:
                print(col_set_iter, question_arg[count_q])
                raise RuntimeError("not in col set")
        elif t == 'agg':
            one_hot_type[count_q][2] = 1
        elif t == 'MORE':
            one_hot_type[count_q][3] = 1
        elif t == 'MOST':
            one_hot_type[count_q][4] = 1
        elif t == 'value':
            one_hot_type[count_q][5] = 1
            question_arg[count_q] = ['value'] + question_arg[count_q]
        else:
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    try:
                        col_set_type[sql['col_set'].index(col_probase)][2] = 5
                        question_arg[count_q] = ['value'] + question_arg[count_q]
                    except:
                        print(sql['col_set'], col_probase)
                        raise RuntimeError('not in col')
                    one_hot_type[count_q][5] = 1
            else:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    col_set_type[sql['col_set'].index(col_probase)][3] += 1


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
        lf.build_tree(copy.copy(rule_label))
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
