import argparse
import pandas as pd
import polars as pl
from kkdf.util.function import load_pickle, get_variance
from kklogger import set_logger


parser = argparse.ArgumentParser(
    description="This command is easy way to check variance of a dataframe.",
    epilog=r"""
    [kkdfvar --df df.pickle --sort race_id,number]
    """
)
parser.add_argument("--df",    type=str, help="dataframe. ex) --df df.pickle", required=True),
parser.add_argument("--ratio", type=float, help="ratio. ex) --ratio 0.95", default=0.95),
parser.add_argument("--disp",  type=int, help="display. ex) --disp 2", default=2),
args   = parser.parse_args()
LOGGER = set_logger(__name__)


def check_variance(args=args, list_df: list[pd.DataFrame | pl.DataFrame] = None):
    LOGGER.info(f"{args}")
    assert list_df is None or list_df == []
    df, _ = load_pickle(args.df)
    get_variance(df, check_ratio=args.ratio, n_display=args.disp)
    if list_df is not None:
        list_df.append(df)


if __name__ == "__main__":
    list_df = []
    df = check_variance(args=args, list_df=list_df)
    if list_df is not None:
        list_df.append(df)
