# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/24
# @Author  : Jiaqi&Zecheng
# @File    : sql2SemQL.py
# @Software: PyCharm
"""

import argparse
import json
import sys
from typing import List, Union

from preprocessing.utils import load_dataSets, find_table_of_star_column
from intermediate_representation.semQL import Root1, Root, N, A, C, T, Sel, Sup, Filter, Order, V

sys.path.append("..")


class Parser:

    def __init__(self, values=None, build_value_list: bool = False) -> None:
        """
        The SQL to SemQL parser will transform SQL to SemQL. It can handle all supported features of SemQL.
        The "values" parameter is a list of values from which the Parser will incorporate the correct one in the query.
        If a value is not provided but required during parsing, it will result in an exception.

        If the build_value_list parameter is provided, the Parser will build the values-list up on the fly and choose the value itself.
        @param values:
        @param build_value_list
        """
        if values is not None:
            # this values will get used to create the SemQL ground truth - for each value, the index to that value will be saved in the V-action (e.g. V(5))
            self.values = values
        else:
            self.values = []

        self.build_value_list = build_value_list

    def _parse_root(self, sql):
        """
        parsing the sql by the grammar
        R ::= Select | Select Filter | Select Order | ... |
        :return: [R(), states]
        """
        use_sup, use_ord, use_fil = True, True, False

        if sql['sql']['limit'] == None:
            use_sup = False

        if sql['sql']['orderBy'] == []:
            use_ord = False
        elif sql['sql']['limit'] != None:
            use_ord = False

        # check the where and having
        if sql['sql']['where'] != [] or \
                sql['sql']['having'] != []:
            use_fil = True

        if use_fil and use_sup:
            return [Root(0)], ['FILTER', 'SUP', 'SEL']
        elif use_fil and use_ord:
            return [Root(1)], ['ORDER', 'FILTER', 'SEL']
        elif use_sup:
            return [Root(2)], ['SUP', 'SEL']
        elif use_fil:
            return [Root(3)], ['FILTER', 'SEL']
        elif use_ord:
            return [Root(4)], ['ORDER', 'SEL']
        else:
            return [Root(5)], ['SEL']

    @staticmethod
    def _parser_column0(sql, select):
        table_idx = find_table_of_star_column(sql, select)
        return T(table_idx)

    def _parse_select(self, sql):
        """
        parsing the sql by the grammar
        Select ::= A | AA | AAA | ... |
        A ::= agg column table
        :return: [Sel(), states]
        """
        result = []
        is_distinct = sql['sql']['select'][0]  # is distinct on the whole select.
        select = sql['sql']['select'][1]

        # as a simplification we assume that if any of the columns is distinct, the whole query is distinct. This might be oversimplified, but
        # it is actually hard to find a way to phrase a real question where some columns are distinct and others not. And in the DEV set, there is also no such example, so we
        # simplified the SemQL language to that.
        if not is_distinct:
            is_distinct = any(sel[1][1][2] for sel in select)

        if is_distinct:
            result.append(Sel(1))
        else:
            result.append(Sel(0))

        result.append(N(len(select) - 1))  # N() encapsulates the number of columns. The -1 is used in case there is only one column to select: in that case, #0 of grammar_dict is used, which is 'N A'.

        for sel in select:
            result.append(A(sel[0])) # A() represents an aggregator. e.g. #0 is 'none', #3 is 'count'
            result.append(C(sql['col_set'].index(sql['names'][sel[1][1][1]])))
            # now check for the situation with *
            if sel[1][1][1] == 0:
                result.append(self._parser_column0(sql, select))  # The "*" needs an extra handling, as it belongs not to a "normal" table.
            else:
                result.append(T(sql['col_table'][sel[1][1][1]]))  # for every other column, we can simply add a T() with the table this column belongs to.

        return result, None

    def _parse_sup(self, sql):
        """
        parsing the sql by the grammar
        Sup ::= Most A | Least A
        A ::= agg column table
        :return: [Sup(), states]
        """
        result = []
        select = sql['sql']['select'][1]
        if sql['sql']['limit'] == None:
            return result, None
        if sql['sql']['orderBy'][0] == 'desc':
            result.append(Sup(0))
        else:
            result.append(Sup(1))

        result.append(A(sql['sql']['orderBy'][1][0][1][0]))
        result.append(C(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]])))
        if sql['sql']['orderBy'][1][0][1][1] == 0:
            result.append(self._parser_column0(sql, select))
        else:
            result.append(T(sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]))
        return result, None

    def _parse_filter(self, sql):
        """
        parsing the sql by the grammar
        Filter ::= and Filter Filter | ... |
        A ::= agg column table
        :return: [Filter(), states]
        """
        result = []

        # in case there are WHERE and HAVING restrictions, we concatenate them with an AND
        if sql['sql']['where'] != [] and sql['sql']['having'] != []:
            result.append(Filter(0))

        if sql['sql']['where']:
            # the real magic happens in this recursive function
            result.extend(self._parse_filter_internal(sql['sql']['where'], sql))

        if sql['sql']['having']:
            # the real magic happens in this recursive function
            result.extend(self._parse_filter_internal(sql['sql']['having'], sql))
        return result, None

    def _parse_filter_internal(self, filter_conditions: List[Union[dict, str]], sql: dict) -> List[Filter]:
        """
        This recursive method is parsing the whole list of filters. It will divide the list by "AND" and "OR" operators
        until only one condition is left, which will be parsed.
        """
        if len(filter_conditions) >= 3:
            results = []

            # TODO: The order of AND and OR here is essential. In my opinion it should be "OR" first, but because the
            # TODO: SemQL2SQL module can't handle it right now, we leave it like this. See comments in test test_full_parse__four_AND_one_OR()
            if 'and' in filter_conditions:
                results.append(Filter(0))
                idx_condition = filter_conditions.index('and')
            elif 'or' in filter_conditions:
                results.append(Filter(1))
                idx_condition = filter_conditions.index('or')
            else:
                raise ValueError('if there are 3 where clauses left, there need to be either an AND or an OR to concatenate them!')

            results.extend(self._parse_filter_internal(filter_conditions[:idx_condition], sql))
            results.extend(self._parse_filter_internal(filter_conditions[idx_condition + 1:], sql))

            return results

        if len(filter_conditions) == 1:
            return self.parse_one_condition(filter_conditions[0], sql['names'], sql)

        raise ValueError('There should always be an uneven amount of filter conditions (e.g. [cond_1, "AND", cond2]. Check "where"/"having" clause')

    def _parse_order(self, sql):
        """
        parsing the sql by the grammar
        Order ::= asc A | desc A
        A ::= agg column table
        :return: [Order(), states]
        """
        result = []

        if 'order' not in sql['query_toks_no_value'] or 'by' not in sql['query_toks_no_value']:
            return result, None
        elif 'limit' in sql['query_toks_no_value']:
            return result, None
        else:
            if sql['sql']['orderBy'] == []:
                return result, None
            else:
                select = sql['sql']['select'][1]
                if sql['sql']['orderBy'][0] == 'desc':
                    result.append(Order(0))
                else:
                    result.append(Order(1))
                result.append(A(sql['sql']['orderBy'][1][0][1][0]))
                result.append(C(sql['col_set'].index(sql['names'][sql['sql']['orderBy'][1][0][1][1]])))
                if sql['sql']['orderBy'][1][0][1][1] == 0:
                    result.append(self._parser_column0(sql, select))
                else:
                    result.append(T(sql['col_table'][sql['sql']['orderBy'][1][0][1][1]]))
        return result, None

    def parse_one_condition(self, sql_condit, names, sql):
        result = []
        # check if V(root)
        nest_query = True
        if type(sql_condit[3]) != dict:
            nest_query = False

        if sql_condit[0] == True:
            if sql_condit[1] == 9:
                # not like only with values
                fil = Filter(10)
            elif sql_condit[1] == 8:
                # not in with Root
                fil = Filter(19)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")
        else:
            # check for Filter (<,=,>,!=,between, >=,  <=, ...)
            # Ursin: This map is a mapping between the index of the WHERE_OPS in spider and the Filter() index in SemQL:
            # WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
            # Filter --> see Filter-class
            # Example: 1:8 --> the filter type "between" is a 1 in the spider notation, but a 8 in SemQL.
            single_map = {1: 8, 2: 2, 3: 5, 4: 4, 5: 7, 6: 6, 7: 3}
            nested_map = {1: 15, 2: 11, 3: 13, 4: 12, 5: 16, 6: 17, 7: 14}
            if sql_condit[1] in [1, 2, 3, 4, 5, 6, 7]:
                if nest_query == False:
                    fil = Filter(single_map[sql_condit[1]])
                else:
                    fil = Filter(nested_map[sql_condit[1]])
            elif sql_condit[1] == 9:
                fil = Filter(9)
            elif sql_condit[1] == 8:
                fil = Filter(18)
            else:
                print(sql_condit[1])
                raise NotImplementedError("not implement for the others FIL")

        result.append(fil)

        result.append(A(sql_condit[2][1][0]))
        result.append(C(sql['col_set'].index(sql['names'][sql_condit[2][1][1]])))
        if sql_condit[2][1][1] == 0:
            select = sql['sql']['select'][1]
            result.append(self._parser_column0(sql, select))
        else:
            result.append(T(sql['col_table'][sql_condit[2][1][1]]))

        # This are filter statements which contain Values - we extend the SemQL AST with a "V" action and use the index
        # of the value based on the provided value list (important: the value needs to exist in the list!)
        if 2 <= fil.id_c <= 10:
            val = sql_condit[3]
            value_action = self._build_value_action(val)
            result.append(value_action)

            # Filter(8) is the "X.Y BETWEEN A AND B" case - here we have to store an additional value.
            if fil.id_c == 8:
                val = sql_condit[4]
                value_action = self._build_value_action(val)
                result.append(value_action)

        # check for the nested value
        if type(sql_condit[3]) == dict:
            nest_query = {}
            nest_query['names'] = names
            nest_query['query_toks_no_value'] = ""
            nest_query['sql'] = sql_condit[3]
            nest_query['col_table'] = sql['col_table']
            nest_query['col_set'] = sql['col_set']
            nest_query['table_names'] = sql['table_names']
            nest_query['question'] = sql['question']
            nest_query['query'] = sql['query']
            nest_query['keys'] = sql['keys']
            result.extend(self.parser(nest_query))

        return result

    def _parse_step(self, state, sql):

        if state == 'ROOT':
            return self._parse_root(sql)

        if state == 'SEL':
            return self._parse_select(sql)

        elif state == 'SUP':
            return self._parse_sup(sql)

        elif state == 'FILTER':
            return self._parse_filter(sql)

        elif state == 'ORDER':
            return self._parse_order(sql)
        else:
            raise NotImplementedError("Not the right state")

    def _build_value_action(self, val):
        # the ground truth values are often in a weird format (e.g. 56.0 instead of 56) and as we anyway only deal with string here,
        # we format every value as a simple string. This also simplifies comparison in the further code. When writing the query, we will
        # adapt it again (see sem2SQL.py --> format_value_given_datatype()).
        val = self.format_groundtruth_value(val)

        # when using the Parser with this parameter = True, we automatically build up the value_list during parsing.
        if self.build_value_list:
            if val not in self.values:
                self.values.append(val)

        try:
            return V(self.values.index(val))
        except ValueError as e:
            raise ValueError(
                f'could not find value "{val}" in the provided list of values "{self.values}". '
                f'Make sure all necessary values are provided in the constructor of the parser or use the parser with parameter "build_value_list" = True') from e

    @staticmethod
    def format_groundtruth_value(val):
        if isinstance(val, str):
            val = val.strip('\'\"')  # remove string quotes, as we will add them later anyway.

            # if it is a fuzzy string (e.g. '%hello%') we wan't to remove the wildcards, as they get in later as part of the post-processing.
            if val.startswith('%') or val.endswith('%'):
                val = val.replace('%', '')

            # some ground truth values contain a trailing tab - no idea why.
            if val.endswith('\t'):
                val = val.rstrip()

        # the ground truth values are all floats, even if there is no decimals (e.g. 56.0 instead of 56). But to make the
        # .index() work, we need exact matches!
        if isinstance(val, float) and val.is_integer():
            val = int(val)

        # results from NER will only be strings - therefore we need to make sure the values we use here are also string only!
        return str(val)

    def full_parse(self, query):
        """
        With this code we pare a SQL-query (as specified by spider) into a SemQL-AST
        """

        sql = query['sql']

        nest_query = {}
        nest_query['names'] = query['names']
        nest_query['query_toks_no_value'] = ""
        nest_query['col_table'] = query['col_table']
        nest_query['col_set'] = query['col_set']
        nest_query['table_names'] = query['table_names']
        nest_query['question'] = query['question']
        nest_query['query'] = query['query']
        nest_query['keys'] = query['keys']

        ### the following 3 if's are exception case, for "intersect", "union" and "except" cases. A normal query uses the [Root1(3)] and calls self.parser only once ###

        if sql['intersect']:
            results = [Root1(0)]
            nest_query['sql'] = sql['intersect']
            # an intersect query, similar to "union" and "except", will call the parser twice (once for each sub-query)
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['union']:
            results = [Root1(1)]
            nest_query['sql'] = sql['union']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        if sql['except']:
            results = [Root1(2)]
            nest_query['sql'] = sql['except']
            results.extend(self.parser(query))
            results.extend(self.parser(nest_query))
            return results

        results = [Root1(3)]
        results.extend(self.parser(query))

        return results

    def parser(self, query):
        stack = ["ROOT"]
        result = []
        while len(stack) > 0:
            state = stack.pop()
            step_result, step_state = self._parse_step(state, query)
            result.extend(step_result)
            if step_state:
                stack.extend(step_state)
        return result


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, help='dataset', required=True)
    arg_parser.add_argument('--table_path', type=str, help='table dataset', required=True)
    arg_parser.add_argument('--output', type=str, help='output data', required=True)
    arg_parser.add_argument('--use_ner_value_candidates', action='store_true', default=True, help="we can either let the model predict from the ground truth-values only (values extracted directly from the SQL-structure) "
                                                                                                  "or we can instead let it predict the right value from a set of possible values extracted by NER and handcrafted heuristics (see pre_process_ner_values.py)")
    args = arg_parser.parse_args()

    # loading dataSets
    data, table = load_dataSets(args)
    processed_data = []

    with open('new.txt', 'a') as the_file:
        for row in data:
            if len(row['sql']['select'][1]) > 5:
                print('more than 5 rows to select! Currently not implemented')
                continue

            # we can either let the model predict from the ground truth-values only (values extracted directly from the SQL-structure) or we can instead
            # let it predict the right value from a set of possible values extracted by NER and handcrafted heuristics (see pre_process_ner_values.py)
            if args.use_ner_value_candidates:
                parser = Parser(row['ner_extracted_values_processed'])

                # When using the NER-values, we override here the ground truth values with it. That way the training/evaluation scripts stay
                # exactly the same and using values/ner-extracted values is decided here. To make this clear we also remove all other value lists.
                # Be aware that at this point, we already have added missing values from the ground-truth if necessary (see pre-processing and "all_values_found" flag)
                row['values'] = row['ner_extracted_values_processed']
                del row['ner_extracted_values_processed']
            else:
                parser = Parser(row['values'])

            # here is where the magic happens: we parse the SQL from the spider-examples and create a SemQL-AST fro it.
            semql_result = parser.full_parse(row)

            # here we simply serialize it to a string. Keep in mind that the SemQL-Classes (e.g. "Root") override the string method, so we get not only the class
            # but also all attributes (especially the idx of the production rule)
            row['rule_label'] = " ".join([str(x) for x in semql_result])

            the_file.write(row['rule_label'] + '\n')

            print(row['rule_label'])

            processed_data.append(row)

    print('Finished %s datas and failed %s datas' % (len(processed_data), len(data) - len(processed_data)))
    with open(args.output, 'w', encoding='utf8') as f:
        f.write(json.dumps(processed_data, indent=2))
