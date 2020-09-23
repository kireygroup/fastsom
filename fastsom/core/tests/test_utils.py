import unittest

from nose2.tools import params
from fastsom.core import ifnone, is_iterable, listify, setify, compose


class IfNoneTest(unittest.TestCase):
    @params(5, [1, 2, 3, 4], "hello world", {"a": 1})
    def test_element(self, el):
        assert ifnone(el, None) is not None

    @params(5, [1, 2, 3, 4], "hello world", {"a": 1})
    def test_none(self, default):
        assert ifnone(None, default) is not None


class ListifyTest(unittest.TestCase):
    def test_list(self):
        l = [1, 2, 3, 4]
        assert l == listify(l)

    def test_dict(self):
        d = {"a": 1, "b": 2, "c": 3}
        assert ["a", "b", "c"] == listify(d)

    def test_set(self):
        s = {1, 2, 3}
        assert [1, 2, 3] == listify(s)

    def test_none(self):
        l = None
        assert [] == listify(l)

    def test_string(self):
        s = "hello world"
        assert list(s) == listify(s)

    def test_value(self):
        assert 5 == listify(5)


class SetifyTest(unittest.TestCase):
    def test_list(self):
        l = [1, 2, 3, 4]
        assert {1, 2, 3, 4} == setify(l)

    def test_dict(self):
        d = {"a": 1, "b": 2, "c": 3}
        assert {"a", "b", "c"} == setify(d)

    def test_none(self):
        assert set() == setify(None)

    def test_string(self):
        s = "hello world"
        assert set(list(s)) == setify(s)

    @params(5, 150.0)
    def test_value(self, value):
        assert value == setify(value)


class IsIterableTest(unittest.TestCase):
    @params(
        (5, False),
        (150.0, False),
        ("Hello world", True),
        ((1, 2,), True),
        ([1, 2], True),
        ({"a": 1, "b": 2}, True),
        ({1, 2, 3}, True),
    )
    def test_is_iterable(self, value: any, expected: bool):
        assert expected == is_iterable(value)


class ComposeTest(unittest.TestCase):
    def test_compose(self):
        from itertools import repeat

        fns = repeat(lambda x: x * 2, 8)
        assert 512 == compose(2, fns)
