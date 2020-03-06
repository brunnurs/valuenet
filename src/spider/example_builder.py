import copy
import numpy as np
from nltk import WordNetLemmatizer

from intermediate_representation import lf
from spider.example import Example

# Take care, this imports are necessary due to the dynamic "eval()" command further down
from intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, Root1, V

wordnet_lemmatizer = WordNetLemmatizer()


def build_example(sql, table_data):
    # The table data contains detailed information about that database. This includes tables, table headers (columns)
    # and also foreign/primary keys.
    table = table_data[sql['db_id']]

    process_dict = _process(sql, table)

    for c_id, col_ in enumerate(process_dict['col_set_iter']):
        for q_id, ori in enumerate(process_dict['q_iter_small']):
            if ori in col_:
                # if we have a match between a partial column token (e.g. "horse id") and a token in the question (e.g. "horse")
                # we will increase the counter for this column
                process_dict['col_set_type'][c_id][0] += 1

    _schema_linking(process_dict['question_arg'], process_dict['question_arg_type'],
                    process_dict['one_hot_type'], process_dict['col_set_type'], process_dict['col_set_iter'], sql)

    # this dict is telling for each column in what table it appears. So the key is the idx of the column, the values the idx of the tables.
    # example: the key 0 will appear in all tables (e.g. [1,2,3,4,5]) as it's the special column "*".
    # Most others will only appear one, but the id's (which are used as primary key / foreign key) will appear also multiple times.
    col_table_dict = _get_col_table_dict(process_dict['tab_cols'], process_dict['tab_ids'], sql)
    # a simple list with sublists for each table, containing all the columns in that table.
    table_col_name = _get_table_colNames(process_dict['tab_ids'], process_dict['col_iter'])

    # this field contains the special column "*", referring to all columns in all tables. Not sure yet why we replace it with this special content.
    process_dict['col_set_iter'][0] = ['count', 'number', 'many']

    # in the pre-processing (see sql2SemQL.py) we parse the sql for each example to the SemQL-AST language. We then serialize it to a string.
    # Here we do the opposite: we deserialize the SemQL-Query by dynamically create the right objects based on the string.
    rule_label = None
    if 'rule_label' in sql:
        # Example: eval("Root1(3)") will dynamically create an instance of class Root1 with the constructor argument 3.
        rule_label = [eval(x) for x in sql['rule_label'].strip().split(' ')]

        if is_valid(rule_label, col_table_dict=col_table_dict, sql=sql) is False:
            raise RuntimeError("Invalid rule_label: {}. We don't use this sample".format(sql['rule_label']))

    # For details about the following values, see the constructor documentation of the "Example" class.
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
        tgt_actions=rule_label,
        values=sql['values']
    )
    example.sql_json = copy.deepcopy(sql)

    return example


def _process(sql, table):
    """
    In this method we further pre-process the data. We for example split colum names in single words. and create new
    one-hot-encoded vectors for the question token types (one_hot_type)
    """
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
    """
    Schema linking. Be aware that a large part of the job is done in  the pre-processing. What we do here is mostly putting all information together in the "one_hot_type" array,
    to use this later to learn from. All this heuristic information are basically used to bootstrap the deep learning approach, where the model gets this encoded information and can
    decide what to learn from it.

    The arguments bellow are from a real question.
    @param question_arg: [['what'], ['is'], ['official', 'name'], ['and'], ['status'], ['of'], ['city'], ['with'], ['most'], ['resident'], ['?']]
    @param question_arg_type: [['NONE'], ['NONE'], ['col'], ['NONE'], ['col'], ['NONE'], ['table'], ['NONE'], ['agg'], ['NONE'], ['NONE']]
    @param one_hot_type: this array we fill in this method. It will contain for each question token a one-hot-encoded "type", which can e.g. be "column", "table", "MORE", etc.
    @param col_set_type:
    @param col_set_iter: [['*'], ['city', 'id'], ['official', 'name'], ['status'], ['area', 'km', '2'], ['population'], ['census', 'ranking'], ['farm', 'id'], ['year'], ['total', 'horse'], ['working', 'horse'], ['total', 'cattle'], ['ox'], ['bull'], ['cow'], ['pig'], ['sheep', 'and', 'goat'], ['competition', 'id'], ['theme'], ['host', 'city', 'id'], ['host'], ['rank']]
    @param sql:
    """
    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]

        # go to the pre-processing (data_process.py) to understand the types ('col', 'table', 'MORE', etc.) better
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            question_arg[count_q] = ['table'] + question_arg[
                count_q]  # we also add the information straight before the question token (e.g. [['in'],['table', 'horse']])
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            try:
                # to my understanding we want to indicate with this that there is an exact match with a column. The col_set_type at #0 contains a value for partial matches (+1 for every matching token)
                # so the +5 at #1 will most probably be more weight than a partial match can be. The exact reason for +5 is though unknown to me.
                col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                question_arg[count_q] = ['column'] + question_arg[count_q]  # we also add the information straight before the question token (e.g. [['are'],['column', 'name']])
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
            question_arg[count_q] = ['value'] + question_arg[
                count_q]  # we also add the information straight before the question token (e.g. [['then'],['value', '5000']])
        else:
            # this are special cases, where "col_probase" is a value we found with ConeceptNet. So "col_probase" could for example be "time of day" or "city"
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    try:
                        # if the token type (col_probase) is e.g. "time of day" (because the word was "night" and we found with ConceptNet it is a "time of day")
                        # we try to find a column with this name. If we find it --> we have an "Exact match", so we mark the matching column with a 5 (in the "col_set_type" array)
                        # NOTE: we could do this much smarter by using the table values.
                        col_set_type[sql['col_set'].index(col_probase)][2] = 5
                        # we also know the token is a "value"
                        question_arg[count_q] = ['value'] + question_arg[count_q]
                    except:
                        print(sql['col_set'], col_probase)
                        raise RuntimeError('not in col')
                    one_hot_type[count_q][5] = 1  # it is a "value" as well.
            else:
                for col_probase in t_q:  # if there are multiple types, we try to give partial matches to the according columns. Not sure there is such cases with this pre-processing though...
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
