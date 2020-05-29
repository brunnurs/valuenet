import copy
import json

import numpy as np

from intermediate_representation.semQL import Sup, Sel, Order, Root, Filter, A, N, C, T, V, Root1


def _build_single_filter(lf, f):
    # No conjunction
    agg = lf.pop(0)
    column = lf.pop(0)
    if len(lf) == 0:
        table = None
    else:
        table = lf.pop(0)
        if not isinstance(table, T):
            lf.insert(0, table)
            table = None
    assert isinstance(agg, A) and isinstance(column, C)

    # This subset includes all "normal" filter, so no concatenation (AND/OR)
    # or subqueries. See Filter() grammar
    if 2 <= f.id_c <= 10:
        f.add_children(agg)
        agg.set_parent(f)
        agg.add_children(column)
        column.set_parent(agg)
        if table is not None:
            column.add_children(table)
            table.set_parent(column)

        value1 = lf.pop(0)
        value1.set_parent(f)
        f.add_children(value1)

        # there is currently one filter type (8, BETWEEN) which contains two values.
        if len(lf) > 0 and isinstance(lf[0], V):
            value2 = lf.pop(0)
            value2.set_parent(f)
            f.add_children(value2)

    else:
        # Subquery
        f.add_children(agg)
        agg.set_parent(f)
        agg.add_children(column)
        column.set_parent(agg)
        if table is not None:
            column.add_children(table)
            table.set_parent(column)
        _root = _build(lf)
        f.add_children(_root)
        _root.set_parent(f)


def _build_filter(lf, root_filter):
    assert isinstance(root_filter, Filter)
    op = root_filter.production.split()[1]
    if op == 'and' or op == 'or':
        for i in range(2):
            child = lf.pop(0)
            op = child.production.split()[1]
            if op == 'and' or op == 'or':
                _f = _build_filter(lf, child)
                root_filter.add_children(_f)
                _f.set_parent(root_filter)
            else:
                _build_single_filter(lf, child)
                root_filter.add_children(child)
                child.set_parent(root_filter)
    else:
        _build_single_filter(lf, root_filter)
    return root_filter


def _build(lf):
    root = lf.pop(0)
    assert isinstance(root, Root)
    length = len(root.production.split()) - 1
    while len(root.children) != length:
        c_instance = lf.pop(0)
        if isinstance(c_instance, Sel):
            sel_instance = c_instance
            root.add_children(sel_instance)
            sel_instance.set_parent(root)

            # define_rule.N
            c_instance = lf.pop(0)
            c_instance.set_parent(sel_instance)
            sel_instance.add_children(c_instance)
            assert isinstance(c_instance, N)
            for i in range(c_instance.id_c + 1):
                agg = lf.pop(0)
                column = lf.pop(0)
                if len(lf) == 0:
                    table = None
                else:
                    table = lf.pop(0)
                    if not isinstance(table, T):
                        lf.insert(0, table)
                        table = None
                assert isinstance(agg, A) and isinstance(column, C)
                c_instance.add_children(agg)
                agg.set_parent(c_instance)
                agg.add_children(column)
                column.set_parent(agg)
                if table is not None:
                    column.add_children(table)
                    table.set_parent(column)

        elif isinstance(c_instance, Sup) or isinstance(c_instance, Order):
            root.add_children(c_instance)
            c_instance.set_parent(root)

            agg = lf.pop(0)
            column = lf.pop(0)
            if len(lf) == 0:
                table = None
            else:
                table = lf.pop(0)
                if not isinstance(table, T):
                    lf.insert(0, table)
                    table = None
            assert isinstance(agg, A) and isinstance(column, C)
            c_instance.add_children(agg)
            agg.set_parent(c_instance)
            agg.add_children(column)
            column.set_parent(agg)
            if table is not None:
                column.add_children(table)
                table.set_parent(column)

        elif isinstance(c_instance, Filter):
            _build_filter(lf, c_instance)
            root.add_children(c_instance)
            c_instance.set_parent(root)

    return root


def build_tree(lf):
    root = lf.pop(0)
    assert isinstance(root, Root1)
    if root.id_c == 0 or root.id_c == 1 or root.id_c == 2:
        root_1 = _build(lf)
        root_2 = _build(lf)
        root.add_children(root_1)
        root.add_children(root_2)
        root_1.set_parent(root)
        root_2.set_parent(root)
    else:
        root_1 = _build(lf)
        root.add_children(root_1)
        root_1.set_parent(root)
    verify(root)
    # eliminate_parent(root)


def eliminate_parent(node):
    for child in node.children:
        eliminate_parent(child)
    node.children = list()


def verify(node):
    if isinstance(node, C) and len(node.children) > 0:
        table = node.children[0]
        assert table is None or isinstance(table, T)
    if isinstance(node, T):
        return
    children_num = len(node.children)
    if isinstance(node, Root1):
        if node.id_c == 0 or node.id_c == 1 or node.id_c == 2:
            assert children_num == 2
        else:
            assert children_num == 1
    elif isinstance(node, Root):
        assert children_num == len(node.production.split()) - 1
    elif isinstance(node, N):
        assert children_num == int(node.id_c) + 1
    elif isinstance(node, Sup) or isinstance(node, Order) or isinstance(node, Sel):
        assert children_num == 1
    elif isinstance(node, Filter):
        op = node.production.split()[1]
        if op == 'and' or op == 'or':
            assert children_num == 2
        else:
            # This subset includes all "normal" filter, so no concatenation (AND/OR)
            # or subqueries. See Filter() grammar
            if 2 <= node.id_c <= 10:
                if node.id_c == 8:
                    assert children_num == 3
                else:
                    assert children_num == 2
            else:
                assert children_num == 2
    for child in node.children:
        assert child.parent == node
        verify(child)


def label_matrix(lf, matrix, node):
    nindex = lf.index(node)
    for child in node.children:
        if child not in lf:
            continue
        index = lf.index(child)
        matrix[nindex][index] = 1
        label_matrix(lf, matrix, child)


def build_adjacency_matrix(lf, symmetry=False):
    _lf = list()
    for rule in lf:
        if isinstance(rule, A) or isinstance(rule, C) or isinstance(rule, T):
            continue
        _lf.append(rule)
    length = len(_lf)
    matrix = np.zeros((length, length,))
    label_matrix(_lf, matrix, _lf[0])
    if symmetry:
        matrix += matrix.T
    return matrix


if __name__ == '__main__':
    with open(r'data/spider/train.json', 'r', encoding='utf-8') as file_handle:
        data = json.load(file_handle)
    for d in data:
        rule_label = [eval(x) for x in d['rule_label'].strip().split(' ')]
        print(d['question'])
        print(rule_label)
        build_tree(copy.copy(rule_label))
        adjacency_matrix = build_adjacency_matrix(rule_label, symmetry=True)
        print(adjacency_matrix)
        print('===\n\n')
