import pytest
from kkdf.util.com import check_type, check_type_list


class TestCheckType:
    def test_single_type_match(self):
        assert check_type(1, int) is True
        assert check_type("hello", str) is True
        assert check_type(1.0, float) is True
        assert check_type(True, bool) is True
        assert check_type([], list) is True
        assert check_type({}, dict) is True

    def test_single_type_no_match(self):
        assert check_type(1, str) is False
        assert check_type("hello", int) is False
        assert check_type(1.0, int) is False

    def test_multiple_types_match(self):
        assert check_type(1, [int, str]) is True
        assert check_type("hello", [int, str]) is True
        assert check_type(1.0, [int, float]) is True

    def test_multiple_types_no_match(self):
        assert check_type(1.0, [str, list]) is False
        assert check_type([], [int, str]) is False

    def test_tuple_type_list(self):
        assert check_type(1, (int, str)) is True
        assert check_type("hello", (int, str)) is True

    def test_none_type(self):
        assert check_type(None, type(None)) is True
        assert check_type(None, int) is False

    def test_bool_is_int(self):
        # bool is a subclass of int in Python
        assert check_type(True, int) is True


class TestCheckTypeList:
    def test_simple_list_all_match(self):
        assert check_type_list([1, 2, 3, 4], int) is True

    def test_simple_list_no_match(self):
        assert check_type_list([1, 2, 3, "4"], int) is False

    def test_nested_list_match(self):
        assert check_type_list([1, 2, 3, [4, 5]], int, int) is True

    def test_nested_list_no_match(self):
        assert check_type_list([1, 2, 3, [4, 5, 6.0]], int, int) is False

    def test_nested_list_multiple_types(self):
        assert check_type_list([1, 2, 3, [4, 5, 6.0]], int, [int, float]) is True

    def test_empty_list(self):
        assert check_type_list([], int) is True

    def test_single_element(self):
        assert check_type_list([1], int) is True
        assert check_type_list(["a"], str) is True

    def test_non_list_input(self):
        # When instances is not a list/tuple, it falls back to check_type
        assert check_type_list(1, int) is True
        assert check_type_list("hello", str) is True
        assert check_type_list(1, str) is False

    def test_tuple_input(self):
        assert check_type_list((1, 2, 3), int) is True
        assert check_type_list((1, 2, "3"), int) is False

    def test_deeply_nested(self):
        assert check_type_list([1, [2, [3, 4]]], int, int, int) is True
        assert check_type_list([1, [2, [3, "4"]]], int, int, int) is False

    def test_mixed_list_and_non_list(self):
        assert check_type_list([1, 2, [3, 4], 5], int, int) is True

    def test_string_list(self):
        assert check_type_list(["a", "b", "c"], str) is True
        assert check_type_list(["a", "b", 1], str) is False
