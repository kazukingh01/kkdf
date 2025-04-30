import argparse
import pandas as pd
import polars as pl
import numpy as np
from kkdf.util.function import load_pickle
from kklogger import set_logger
from kkdf.util.com import check_type_list


parser = argparse.ArgumentParser(
    description="This command is easy way to concat dataframes.",
    epilog=r"""
    [(1) kkdfconcat df1.pickle df2.pickle --output df_concat.pickle --sort race_id,number]
    [(2) kkdfconcat ./xxxx/yyyy/df_20*.pickle --output df_concat.pickle --sort race_id,number]
    """
)
parser.add_argument("paths",    type=str, nargs="+", help="write dataframe path.")
parser.add_argument("--output", type=str, help="output file name. ex) --output ./df_concat.pickle", default="df_concat.pickle"),
parser.add_argument("--sort",   type=lambda x: x.split(","), help="sort columns. ex) --sort race_id,number"),
args   = parser.parse_args()
LOGGER = set_logger(__name__)


def concat(args=args, list_df: list[pd.DataFrame | pl.DataFrame] = None):
    LOGGER.info(f"{args}")
    assert len(args.paths) >= 2
    assert args.sort is None or check_type_list(args.sort, str)
    assert list_df == []
    ins_type, ignore_index = None, False
    for x in args.paths:
        LOGGER.info(f"load {x} ...")
        if ins_type is None:
            df, ins_type = load_pickle(x)
            if ins_type == "pandas":
                assert args.output.endswith(".pickle")
                if df.index.tolist() == np.arange(df.shape[0], dtype=int).tolist():
                    ignore_index = True
                else:
                    ignore_index = False
            elif ins_type == "polars":
                assert args.output.endswith(".parquet")
                ignore_index = False
        else:
            df, tmp_ins_type = load_pickle(x)
            assert ins_type == tmp_ins_type
        list_df.append(df)
    if ins_type == "pandas":
        df = pd.concat(list_df, axis=0, ignore_index=ignore_index, sort=False)
    elif ins_type == "polars":
        cols = list_df[0].columns
        for fpath, dfwk in zip(args.paths, list_df):
            if dfwk.columns != cols:
                for a in dfwk.columns:
                    if a not in cols:
                        LOGGER.raise_error(f"column: {a} is not in {args.paths[0]}")
                for b in cols:
                    if b not in dfwk.columns:
                        LOGGER.raise_error(f"column: {b} is not in {fpath}")
        df   = pl.concat([dfwk[cols] for dfwk in list_df], how="vertical_relaxed", rechunk=True, parallel=True)
    if args.sort is not None:
        if ins_type == "pandas":
            df = df.sort_values(args.sort, ascending=True)
        elif ins_type == "polars":
            df = df.sort(args.sort, descending=False)
    LOGGER.info(f"save {args.output} ...")
    if ins_type == "pandas":
        df.to_pickle(args.output)
    elif ins_type == "polars":
        df.write_parquet(args.output, compression="zstd")
    if list_df is not None:
        list_df.append(df)


if __name__ == "__main__":
    list_df = []
    concat(args=args, list_df=list_df)
    df = list_df[0]
