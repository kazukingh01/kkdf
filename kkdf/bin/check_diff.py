import argparse
import pandas as pd
import polars as pl
from kkdf.util.function import check_pandas_diff, check_polars_diff, load_pickle
from kklogger import set_logger
from kkdf.util.com import check_type_list


parser = argparse.ArgumentParser(
    description="This command is easy way to check difference between two dataframes.",
    epilog=r"""
    [kkdfdiff --df1 aa.pickle df2 bb.pickle]
    """
)
parser.add_argument("--df1",   type=str, help="dataframe1. ex) --df1 df1.pickle", required=True),
parser.add_argument("--df2",   type=str, help="dataframe1. ex) --df2 df2.pickle", required=True),
parser.add_argument("--index", type=lambda x: x.split(","), help="set index. ex) --index race_id,number"),
parser.add_argument("--fillnan", type=float, help="fill nan value. ex) --fillnan -9999", default=-9999),
args   = parser.parse_args()
LOGGER = set_logger(__name__)


def check_diff(args=args, list_df: list[pd.DataFrame | pl.DataFrame] = None):
    LOGGER.info(f"{args}")
    assert args.index is None or check_type_list(args.index, str)
    assert list_df is None or list_df == []
    df1, ins_type = load_pickle(args.df1)
    df2, ins_type2 = load_pickle(args.df2)
    assert ins_type == ins_type2
    if ins_type == "pandas" and args.index is not None:
        df1 = df1.set_index(args.index)
        df2 = df2.set_index(args.index)
        check_pandas_diff(df1, df2, value_fillnan=args.fillnan)
    elif ins_type == "polars":
        assert args.index is not None
        check_polars_diff(df1, df2, indexes=args.index)
    if list_df is not None:
        list_df.append(df1)
        list_df.append(df2)


if __name__ == "__main__":
    list_df = []
    check_diff(args=args, list_df=list_df)
    df1, df2 = list_df
