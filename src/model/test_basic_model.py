from unittest import TestCase

from model.basic_model import BasicModel

from intermediate_representation.semQL import N, A, C, T, V, Sel, Filter, Order, Sup, Root1, Root


class TestBasicModel(TestCase):
    def test_padding_sketch_filter(self):
        # GIVEN
        sketch_string = "Root1(3) Root(3) Sel(0) N(2) Filter(2)"
        sketch = [eval(x) for x in sketch_string.split(' ')]

        expected_padding_string = "Root1(3) Root(3) Sel(0) N(2) A(0) C(0) T(0) A(0) C(0) T(0) A(0) C(0) T(0) Filter(2) A(0) C(0) T(0) V(0)"

        basic_model = BasicModel()

        # WHEN
        actual_padding = basic_model.padding_sketch(sketch)

        # THEN
        self.assertEqual(expected_padding_string, ' '.join([str(x) for x in actual_padding]))

    def test_padding_sketch_filter__BETWEEN(self):
        # GIVEN
        sketch_string = "Root1(3) Root(3) Sel(0) N(1) Filter(8)"
        sketch = [eval(x) for x in sketch_string.split(' ')]

        expected_padding_string = "Root1(3) Root(3) Sel(0) N(1) A(0) C(0) T(0) A(0) C(0) T(0) Filter(8) A(0) C(0) T(0) V(0) V(0)"

        basic_model = BasicModel()

        # WHEN
        actual_padding = basic_model.padding_sketch(sketch)

        # THEN
        self.assertEqual(expected_padding_string, ' '.join([str(x) for x in actual_padding]))

    def test_padding_sketch__superlative(self):
        # GIVEN
        sketch_string = "Root1(3) Root(2) Sel(0) N(1) Sup(1)"
        sketch = [eval(x) for x in sketch_string.split(' ')]

        expected_padding_string = "Root1(3) Root(2) Sel(0) N(1) A(0) C(0) T(0) A(0) C(0) T(0) Sup(1) A(0) C(0) T(0)"

        basic_model = BasicModel()

        # WHEN
        actual_padding = basic_model.padding_sketch(sketch)

        # THEN
        self.assertEqual(expected_padding_string, ' '.join([str(x) for x in actual_padding]))

    def test_padding_sketch__Root_in_filter(self):
        # GIVEN
        sketch_string = "Root1(3) Root(3) Sel(0) N(0) Filter(13) Root(5) Sel(0) N(0)"
        sketch = [eval(x) for x in sketch_string.split(' ')]

        expected_padding_string = "Root1(3) Root(3) Sel(0) N(0) A(0) C(0) T(0) Filter(13) A(0) C(0) T(0) Root(5) Sel(0) N(0) A(0) C(0) T(0)"

        basic_model = BasicModel()

        # WHEN
        actual_padding = basic_model.padding_sketch(sketch)

        # THEN
        self.assertEqual(expected_padding_string, ' '.join([str(x) for x in actual_padding]))

