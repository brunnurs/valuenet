# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""

from collections import deque, namedtuple


# we'll use infinity as a default distance to nodes.
inf = float('inf')
Edge = namedtuple('Edge', 'start, entry_column, end, exit_column, cost')


def make_edge(start, entry_column, end, exit_column, cost=1):
    return Edge(start, entry_column, end, exit_column, cost)


class Graph:
    def __init__(self, edges):
        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [4, 5]]
        if wrong_edges:
            raise ValueError('Wrong edges data: {}. Format should be: start, entry_column, end, exit_column, cost'.format(wrong_edges))

        self.edges = [make_edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set([e.start for e in self.edges] + [e.end for e in self.edges])

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, source, dest):
        assert source in self.vertices, 'Such source node doesn\'t exist'
        assert dest in self.vertices, 'Such source node doesn\'t exis'

        # 1. Mark all nodes unvisited and store them.
        # 2. Set the distance to zero for our initial node
        # and to infinity for other nodes.
        distances = {vertex: inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()

        while vertices:
            # 3. Select the unvisited node with the smallest distance,
            # it's current node now.
            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])

            # 6. Stop, if the smallest distance
            # among the unvisited nodes is infinity.
            if distances[current_vertex] == inf:
                break

            # 4. Find unvisited neighbors for the current node
            # and calculate their distances through the current node.
            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost

                # Compare the newly calculated distance to the assigned
                # and save the smaller one.
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

            # 5. Mark the current node as visited
            # and remove it from the unvisited set.
            vertices.remove(current_vertex)

        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)

        edges_to_follow = self._get_edges_based_on_path(path)

        return edges_to_follow

    def _get_edges_based_on_path(self, path):
        edges_to_follow = []
        idx = 0
        while idx < len(path) - 1:
            start = path[idx]
            end = path[idx + 1]

            filtered_edges = filter(lambda edge: edge.start == start and edge.end == end, self.edges)
            edges_to_follow.append(next(filtered_edges))

            if next(filtered_edges, None) is not None:
                # TODO: we would need the model to predict the correct relation between multiple tables, if there is more than one. For now we just take the first one we find.
                issue_text = "There should exist exactly one edge between two tables! But we found multiple ones: Tables: {}, {}".format(start, end)
                # print(issue_text)

            idx += 1

        return edges_to_follow


if __name__ == '__main__':
    graph = Graph([("A", "a", "B", "a", 7), ("A", "c", "C", "a", 9), ("A", "f", "F", "a", 14), ("B", "b", "C", "b", 10),
                   ("B", "b", "D", "b", 15), ("C", "c", "D", "c", 11), ("C", "c", "F", "c", 2), ("D", "d", "E", "d", 6),
                   ("E", "e", "F", "e", 9)])

    print(graph.dijkstra("A", "E"))
