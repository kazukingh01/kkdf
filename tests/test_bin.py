import os
import sys
import tempfile
import types
import pytest
import pandas as pd
import polars as pl
import numpy as np


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def make_args(**kwargs):
    return types.SimpleNamespace(**kwargs)


@pytest.fixture(autouse=True)
def patch_sys_argv(monkeypatch):
    """Prevent argparse.parse_args() in bin modules from consuming pytest args."""
    monkeypatch.setattr(sys, "argv", ["test"])


class TestCheckDiff:
    def test_pandas_same(self, tmp_dir):
        from kkdf.bin.check_diff import check_diff
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p1 = os.path.join(tmp_dir, "df1.pickle")
        p2 = os.path.join(tmp_dir, "df2.pickle")
        df1.to_pickle(p1)
        df2.to_pickle(p2)
        args = make_args(df1=p1, df2=p2, index=["a"], fillnan=-9999, unique=False)
        list_df = []
        check_diff(args=args, list_df=list_df)
        assert len(list_df) == 2

    def test_pandas_diff(self, tmp_dir):
        from kkdf.bin.check_diff import check_diff
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 99]})
        p1 = os.path.join(tmp_dir, "df1.pickle")
        p2 = os.path.join(tmp_dir, "df2.pickle")
        df1.to_pickle(p1)
        df2.to_pickle(p2)
        args = make_args(df1=p1, df2=p2, index=["a"], fillnan=-9999, unique=False)
        list_df = []
        check_diff(args=args, list_df=list_df)
        assert len(list_df) == 2

    def test_pandas_no_index(self, tmp_dir):
        from kkdf.bin.check_diff import check_diff
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        p1 = os.path.join(tmp_dir, "df1.pickle")
        p2 = os.path.join(tmp_dir, "df2.pickle")
        df1.to_pickle(p1)
        df2.to_pickle(p2)
        args = make_args(df1=p1, df2=p2, index=None, fillnan=-9999, unique=False)
        list_df = []
        check_diff(args=args, list_df=list_df)
        assert len(list_df) == 2

    def test_pandas_with_index_no_unique(self, tmp_dir):
        from kkdf.bin.check_diff import check_diff
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=pd.Index([10, 20, 30], name="idx"))
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [7, 8, 9]}, index=pd.Index([10, 20, 30], name="idx"))
        p1 = os.path.join(tmp_dir, "df1.pickle")
        p2 = os.path.join(tmp_dir, "df2.pickle")
        df1.to_pickle(p1)
        df2.to_pickle(p2)
        args = make_args(df1=p1, df2=p2, index=None, fillnan=-9999, unique=False)
        list_df = []
        check_diff(args=args, list_df=list_df)
        assert len(list_df) == 2

    def test_polars_same(self, tmp_dir):
        from kkdf.bin.check_diff import check_diff
        df1 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30]})
        p1 = os.path.join(tmp_dir, "df1.parquet")
        p2 = os.path.join(tmp_dir, "df2.parquet")
        df1.write_parquet(p1)
        df2.write_parquet(p2)
        args = make_args(df1=p1, df2=p2, index=["id"], fillnan=-9999, unique=False)
        list_df = []
        check_diff(args=args, list_df=list_df)
        assert len(list_df) == 2

    def test_polars_unique(self, tmp_dir):
        from kkdf.bin.check_diff import check_diff
        df1 = pl.DataFrame({"id": [1, 1, 2], "a": [10, 20, 30]})
        df2 = pl.DataFrame({"id": [1, 2, 2], "a": [40, 50, 60]})
        p1 = os.path.join(tmp_dir, "df1.parquet")
        p2 = os.path.join(tmp_dir, "df2.parquet")
        df1.write_parquet(p1)
        df2.write_parquet(p2)
        args = make_args(df1=p1, df2=p2, index=["id"], fillnan=-9999, unique=True)
        list_df = []
        check_diff(args=args, list_df=list_df)
        assert len(list_df) == 2


class TestCheckVariance:
    def test_pandas(self, tmp_dir):
        from kkdf.bin.check_variance import check_variance
        df = pd.DataFrame({"a": np.random.randint(0, 10, 2000), "b": np.random.rand(2000)})
        path = os.path.join(tmp_dir, "df.pickle")
        df.to_pickle(path)
        args = make_args(df=path, ratio=0.95, disp=2)
        list_df = []
        check_variance(args=args, list_df=list_df)
        assert len(list_df) == 1

    def test_polars(self, tmp_dir):
        from kkdf.bin.check_variance import check_variance
        df = pl.DataFrame({"a": np.random.randint(0, 10, 2000).tolist(), "b": np.random.rand(2000).tolist()})
        path = os.path.join(tmp_dir, "df.parquet")
        df.write_parquet(path)
        args = make_args(df=path, ratio=0.95, disp=2)
        list_df = []
        check_variance(args=args, list_df=list_df)
        assert len(list_df) == 1


class TestConcat:
    def test_pandas_concat(self, tmp_dir):
        from kkdf.bin.concat import concat
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        p1 = os.path.join(tmp_dir, "df1.pickle")
        p2 = os.path.join(tmp_dir, "df2.pickle")
        out = os.path.join(tmp_dir, "out.pickle")
        df1.to_pickle(p1)
        df2.to_pickle(p2)
        args = make_args(paths=[p1, p2], output=out, sort=None, ignore_cols=False)
        list_df = []
        concat(args=args, list_df=list_df)
        # list_df contains: df1, df2, concatenated
        assert len(list_df) == 3
        assert list_df[2].shape[0] == 4
        assert os.path.exists(out)

    def test_pandas_concat_with_sort(self, tmp_dir):
        from kkdf.bin.concat import concat
        df1 = pd.DataFrame({"a": [3, 1], "b": [10, 20]})
        df2 = pd.DataFrame({"a": [2, 4], "b": [30, 40]})
        p1 = os.path.join(tmp_dir, "df1.pickle")
        p2 = os.path.join(tmp_dir, "df2.pickle")
        out = os.path.join(tmp_dir, "out.pickle")
        df1.to_pickle(p1)
        df2.to_pickle(p2)
        args = make_args(paths=[p1, p2], output=out, sort=["a"], ignore_cols=False)
        list_df = []
        concat(args=args, list_df=list_df)
        result = list_df[2]
        assert result["a"].tolist() == [1, 2, 3, 4]

    def test_polars_concat(self, tmp_dir):
        from kkdf.bin.concat import concat
        df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8]})
        p1 = os.path.join(tmp_dir, "df1.parquet")
        p2 = os.path.join(tmp_dir, "df2.parquet")
        out = os.path.join(tmp_dir, "out.parquet")
        df1.write_parquet(p1)
        df2.write_parquet(p2)
        args = make_args(paths=[p1, p2], output=out, sort=None, ignore_cols=False)
        list_df = []
        concat(args=args, list_df=list_df)
        assert len(list_df) == 3
        assert list_df[2].shape[0] == 4
        assert os.path.exists(out)

    def test_polars_concat_with_sort(self, tmp_dir):
        from kkdf.bin.concat import concat
        df1 = pl.DataFrame({"a": [3, 1], "b": [10, 20]})
        df2 = pl.DataFrame({"a": [2, 4], "b": [30, 40]})
        p1 = os.path.join(tmp_dir, "df1.parquet")
        p2 = os.path.join(tmp_dir, "df2.parquet")
        out = os.path.join(tmp_dir, "out.parquet")
        df1.write_parquet(p1)
        df2.write_parquet(p2)
        args = make_args(paths=[p1, p2], output=out, sort=["a"], ignore_cols=False)
        list_df = []
        concat(args=args, list_df=list_df)
        result = list_df[2]
        assert result["a"].to_list() == [1, 2, 3, 4]

    def test_polars_concat_different_cols_error(self, tmp_dir):
        """--ignore-cols なしでカラムが違う場合はエラー"""
        from kkdf.bin.concat import concat
        df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pl.DataFrame({"a": [5, 6], "c": [7, 8]})
        p1 = os.path.join(tmp_dir, "df1.parquet")
        p2 = os.path.join(tmp_dir, "df2.parquet")
        out = os.path.join(tmp_dir, "out.parquet")
        df1.write_parquet(p1)
        df2.write_parquet(p2)
        args = make_args(paths=[p1, p2], output=out, sort=None, ignore_cols=False)
        list_df = []
        with pytest.raises(Exception):
            concat(args=args, list_df=list_df)

    def test_polars_concat_ignore_cols_missing_in_second(self, tmp_dir):
        """df2にカラムが足りない場合、--ignore-cols で null 埋め"""
        from kkdf.bin.concat import concat
        df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        df2 = pl.DataFrame({"a": [7, 8], "b": [9, 10]})
        p1 = os.path.join(tmp_dir, "df1.parquet")
        p2 = os.path.join(tmp_dir, "df2.parquet")
        out = os.path.join(tmp_dir, "out.parquet")
        df1.write_parquet(p1)
        df2.write_parquet(p2)
        args = make_args(paths=[p1, p2], output=out, sort=None, ignore_cols=True)
        concat(args=args, list_df=[])
        result = pl.read_parquet(out)
        assert result.shape == (4, 3)
        assert result.columns == ["a", "b", "c"]
        assert result["c"].to_list() == [5, 6, None, None]

    def test_polars_concat_ignore_cols_extra_in_second(self, tmp_dir):
        """df2に余分なカラムがある場合、--ignore-cols で null 埋め"""
        from kkdf.bin.concat import concat
        df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pl.DataFrame({"a": [5, 6], "b": [7, 8], "c": [9, 10]})
        p1 = os.path.join(tmp_dir, "df1.parquet")
        p2 = os.path.join(tmp_dir, "df2.parquet")
        out = os.path.join(tmp_dir, "out.parquet")
        df1.write_parquet(p1)
        df2.write_parquet(p2)
        args = make_args(paths=[p1, p2], output=out, sort=None, ignore_cols=True)
        concat(args=args, list_df=[])
        result = pl.read_parquet(out)
        assert result.shape == (4, 3)
        assert result.columns == ["a", "b", "c"]
        assert result["c"].to_list() == [None, None, 9, 10]

    def test_polars_concat_ignore_cols_completely_different(self, tmp_dir):
        """共通カラムが一部だけの場合、--ignore-cols で全カラム null 埋め"""
        from kkdf.bin.concat import concat
        df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pl.DataFrame({"a": [5, 6], "c": [7, 8]})
        p1 = os.path.join(tmp_dir, "df1.parquet")
        p2 = os.path.join(tmp_dir, "df2.parquet")
        out = os.path.join(tmp_dir, "out.parquet")
        df1.write_parquet(p1)
        df2.write_parquet(p2)
        args = make_args(paths=[p1, p2], output=out, sort=None, ignore_cols=True)
        concat(args=args, list_df=[])
        result = pl.read_parquet(out)
        assert result.shape == (4, 3)
        assert result.columns == ["a", "b", "c"]
        assert result["b"].to_list() == [3, 4, None, None]
        assert result["c"].to_list() == [None, None, 7, 8]


class TestSplit:
    def test_pandas_random_split(self, tmp_dir):
        from kkdf.bin.split import split
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        path = os.path.join(tmp_dir, "df.pickle")
        df.to_pickle(path)
        args = make_args(df=path, nsplit=5, concat=[[0, 1], [2, 3], [4]], seed=0, splitby=None, ignan=False)
        list_df = []
        split(args=args, list_df=list_df)
        assert len(list_df) == 3
        total = sum(d.shape[0] for d in list_df)
        assert total == 100

    def test_polars_random_split(self, tmp_dir):
        from kkdf.bin.split import split
        df = pl.DataFrame({"a": list(range(100)), "b": list(range(100, 200))})
        path = os.path.join(tmp_dir, "df.parquet")
        df.write_parquet(path)
        args = make_args(df=path, nsplit=5, concat=[[0, 1], [2, 3], [4]], seed=0, splitby=None, ignan=False)
        list_df = []
        split(args=args, list_df=list_df)
        assert len(list_df) == 3
        total = sum(d.shape[0] for d in list_df)
        assert total == 100

    def test_pandas_stratified_split(self, tmp_dir):
        from kkdf.bin.split import split
        np.random.seed(42)
        df = pd.DataFrame({"a": range(100), "class1": np.random.randint(0, 3, 100), "class2": np.random.randint(0, 2, 100)})
        path = os.path.join(tmp_dir, "df.pickle")
        df.to_pickle(path)
        args = make_args(df=path, nsplit=5, concat=[[0, 1, 2, 3], [4]], seed=0, splitby=["class1", "class2"], ignan=False)
        list_df = []
        split(args=args, list_df=list_df)
        assert len(list_df) == 2
        total = sum(d.shape[0] for d in list_df)
        assert total == 100

    def test_polars_stratified_split(self, tmp_dir):
        from kkdf.bin.split import split
        np.random.seed(42)
        df = pl.DataFrame({"a": list(range(100)), "class1": np.random.randint(0, 3, 100).tolist(), "class2": np.random.randint(0, 2, 100).tolist()})
        path = os.path.join(tmp_dir, "df.parquet")
        df.write_parquet(path)
        args = make_args(df=path, nsplit=5, concat=[[0, 1, 2, 3], [4]], seed=0, splitby=["class1", "class2"], ignan=False)
        list_df = []
        split(args=args, list_df=list_df)
        assert len(list_df) == 2
        total = sum(d.shape[0] for d in list_df)
        assert total == 100

    def test_polars_single_splitby(self, tmp_dir):
        from kkdf.bin.split import split
        np.random.seed(42)
        df = pl.DataFrame({"a": list(range(100)), "class1": np.random.randint(0, 3, 100).tolist()})
        path = os.path.join(tmp_dir, "df.parquet")
        df.write_parquet(path)
        args = make_args(df=path, nsplit=3, concat=[[0], [1], [2]], seed=0, splitby=["class1"], ignan=False)
        list_df = []
        split(args=args, list_df=list_df)
        assert len(list_df) == 3
        total = sum(d.shape[0] for d in list_df)
        assert total == 100

    def test_split_output_files_created(self, tmp_dir):
        from kkdf.bin.split import split
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        path = os.path.join(tmp_dir, "df.pickle")
        df.to_pickle(path)
        args = make_args(df=path, nsplit=2, concat=[[0], [1]], seed=0, splitby=None, ignan=False)
        split(args=args, list_df=[])
        assert os.path.exists(f"{path}.split2.0.pickle")
        assert os.path.exists(f"{path}.split2.1.pickle")
