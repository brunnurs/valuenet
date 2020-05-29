from unittest import TestCase

from intermediate_representation.beam import Beams
from intermediate_representation.semQL import Root1, Root, Sel, N, Order, Filter, A, T, C, V


class TestBeams(TestCase):
    def test_get_available_class__filter(self):
        # GIVEN
        hypothesis = Beams(is_sketch=True)
        hypothesis.actions = [Root1(0), Root(4), Sel(1), N(2), Order(1), Root(3), Sel(1), N(1)]

        # WHEN
        available_class = hypothesis.get_availableClass()

        # THEN
        self.assertEqual(Filter, available_class)

    def test_get_available_class__end_of_sketch(self):
        # GIVEN
        hypothesis = Beams(is_sketch=True)
        hypothesis.actions = [Root1(3), Root(5), Sel(1), N(2)]

        # WHEN
        available_class = hypothesis.get_availableClass()

        # THEN
        self.assertIsNone(available_class)

    def test_get_available_class__suggest_A_if_not_sketch(self):
        # GIVEN
        hypothesis = Beams(is_sketch=False)
        hypothesis.actions = [Root1(3), Root(5), Sel(1), N(2)]

        # WHEN
        available_class = hypothesis.get_availableClass()

        # THEN
        self.assertEqual(A, available_class)

    def test_get_available_class__no_values_in_sketch(self):
        # GIVEN
        hypothesis = Beams(is_sketch=True)
        hypothesis.actions = [Root1(3), Root(3), Sel(1), N(2), Filter(2)]

        # WHEN
        available_class = hypothesis.get_availableClass()

        # THEN
        self.assertIsNone(available_class)

    def test_get_available_class__values_included_if_not_sketch(self):
        # GIVEN
        hypothesis = Beams(is_sketch=False)
        hypothesis.actions = [Root1(3), Root(3), Sel(1), N(0), A(0), C(0), T(0), Filter(2), A(0), C(0), T(0)]

        # WHEN
        available_class = hypothesis.get_availableClass()

        # THEN
        self.assertEqual(V, available_class)


