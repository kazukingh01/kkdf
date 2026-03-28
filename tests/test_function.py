import os
import pickle
import tempfile
import pytest
import pandas as pd
import polars as pl
import numpy as np
from kkdf.util.function import load_pickle, check_pandas_diff, check_polars_diff, get_variance


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestLoadPickle:
    def test_load_pandas_pickle(self, tmp_dir):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = os.path.join(tmp_dir, "test.pickle")
        df.to_pickle(path)
        result, ins_type = load_pickle(path)
        assert ins_type == "pandas"
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)

    def test_load_polars_parquet(self, tmp_dir):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = os.path.join(tmp_dir, "test.parquet")
        df.write_parquet(path)
        result, ins_type = load_pickle(path)
        assert ins_type == "polars"
        assert isinstance(result, pl.DataFrame)
        assert result.equals(df)

    def test_load_polars_ipc(self, tmp_dir):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        path = os.path.join(tmp_dir, "test.ipc")
        df.write_ipc(path)
        result, ins_type = load_pickle(path)
        assert ins_type == "polars"
        assert isinstance(result, pl.DataFrame)
        assert result.equals(df)

    def test_load_nonexistent_file(self):
        with pytest.raises(Exception):
            load_pickle("/nonexistent/path/file.pickle")


class TestCheckPandasDiff:
    def test_identical_dataframes(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        r1, r2 = check_pandas_diff(df1, df2)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_values(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 99], "b": [4, 5, 6]})
        r1, r2 = check_pandas_diff(df1, df2)
        assert r1.shape == r2.shape

    def test_with_nan_values(self):
        df1 = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        df2 = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        r1, r2 = check_pandas_diff(df1, df2, value_fillnan=-9999)
        pd.testing.assert_frame_equal(r1, r2)

    def test_with_custom_index(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=["x", "y", "z"])
        r1, r2 = check_pandas_diff(df1, df2)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_columns_subset(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "d": [7, 8]})
        r1, r2 = check_pandas_diff(df1, df2)
        # Should only compare common columns (a, b)
        assert list(r1.columns) == ["a", "b"]
        assert list(r2.columns) == ["a", "b"]

    def test_different_index_subset(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        df2 = pd.DataFrame({"a": [1, 2, 4]}, index=["x", "y", "w"])
        r1, r2 = check_pandas_diff(df1, df2)
        # Should only compare common indices (x, y)
        assert list(r1.index) == ["x", "y"]
        assert list(r2.index) == ["x", "y"]


class TestCheckPolarsDiff:
    def test_identical_dataframes(self):
        df1 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30], "b": [40, 50, 60]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30], "b": [40, 50, 60]})
        r1, r2 = check_polars_diff(df1, df2, indexes=["id"])
        assert r1.equals(r2)

    def test_different_values(self):
        df1 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 99]})
        r1, r2 = check_polars_diff(df1, df2, indexes=["id"])
        assert r1.shape == r2.shape

    def test_different_order(self):
        df1 = pl.DataFrame({"id": [3, 1, 2], "a": [30, 10, 20]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        r1, r2 = check_polars_diff(df1, df2, indexes=["id"])
        assert r1.equals(r2)

    def test_with_null_values(self):
        df1 = pl.DataFrame({"id": [1, 2, 3], "a": [10, None, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "a": [10, None, 30]})
        r1, r2 = check_polars_diff(df1, df2, indexes=["id"])
        assert r1.shape == r2.shape

    def test_multi_index(self):
        df1 = pl.DataFrame({"id1": [1, 2, 3], "id2": ["a", "b", "c"], "val": [10, 20, 30]})
        df2 = pl.DataFrame({"id1": [1, 2, 3], "id2": ["a", "b", "c"], "val": [10, 20, 30]})
        r1, r2 = check_polars_diff(df1, df2, indexes=["id1", "id2"])
        assert r1.equals(r2)

    def test_non_unique_index_raises(self):
        df1 = pl.DataFrame({"id": [1, 1, 2], "a": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        with pytest.raises(AssertionError):
            check_polars_diff(df1, df2, indexes=["id"])

    def test_missing_index_column_raises(self):
        df1 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        with pytest.raises(AssertionError):
            check_polars_diff(df1, df2, indexes=["nonexistent"])


class TestGetVariance:
    def test_pandas_input(self):
        df = pd.DataFrame({
            "a": np.random.randint(0, 100, 2000),
            "b": np.random.rand(2000),
        })
        # Should not raise
        get_variance(df, check_ratio=0.95, n_divide=1000, n_display=3)

    def test_polars_input(self):
        df = pl.DataFrame({
            "a": np.random.randint(0, 100, 2000).tolist(),
            "b": np.random.rand(2000).tolist(),
        })
        get_variance(df, check_ratio=0.95, n_divide=1000, n_display=3)

    def test_with_nan_values(self):
        data = np.random.rand(2000)
        data[::10] = np.nan
        df = pl.DataFrame({"a": data.tolist(), "b": np.random.randint(0, 5, 2000).tolist()})
        get_variance(df, check_ratio=0.95, n_divide=1000, n_display=3)

    def test_low_variance_column(self):
        df = pl.DataFrame({
            "a": [1] * 2000,  # all same value
            "b": list(range(2000)),
        })
        get_variance(df, check_ratio=0.95, n_divide=1000, n_display=3)

    def test_invalid_check_ratio(self):
        df = pl.DataFrame({"a": list(range(2000))})
        with pytest.raises(AssertionError):
            get_variance(df, check_ratio=0.0)
        with pytest.raises(AssertionError):
            get_variance(df, check_ratio=1.0)
        with pytest.raises(AssertionError):
            get_variance(df, check_ratio=1.5)

    def test_invalid_n_divide(self):
        df = pl.DataFrame({"a": list(range(2000))})
        with pytest.raises(AssertionError):
            get_variance(df, n_divide=999)

    def test_invalid_n_display(self):
        df = pl.DataFrame({"a": list(range(2000))})
        with pytest.raises(AssertionError):
            get_variance(df, n_display=0)
