import argparse, random
import pandas as pd
import polars as pl
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from kkdf.util.function import load_pickle
from kkdf.util.com import check_type_list
from kklogger import set_logger


parser = argparse.ArgumentParser(
    description="This command is easy way to split a dataframe and save it.",
    epilog=r"""
    [kkdfsplit --df df.pickle --nsplit 12 --concat 0+1+2+3,4+5,6,7,10 --seed 0 --splitby class1,class2]
    """
)
parser.add_argument("--df",      type=str, help="dataframe. ex) --df df.pickle", required=True),
parser.add_argument("--nsplit",  type=int, help="number of split. ex) --nsplit 5", required=True),
parser.add_argument("--concat",  type=lambda x: [[int(z) for z in y.split("+")] for y in x.split(",")], help="concat for spliting validation and save them. ex) --concat 0123,45,6,7", required=True),
parser.add_argument("--seed",    type=int, help="random seed. ex) --seed 0", default=0),
parser.add_argument("--splitby", type=lambda x: x.split(","), help="split by. ex) --splitby class1,class2", default=None),
parser.add_argument("--ignan",   action="store_true", help="ignore nan when 'splitby' is specified. ex) --ignan", default=False),
args   = parser.parse_args()
LOGGER = set_logger(__name__)
random.seed(args.seed)
np.random.seed(args.seed)


def split(args=args, list_df: list[pd.DataFrame | pl.DataFrame] = None):
    LOGGER.info(f"{args}")
    assert list_df is None or list_df == []
    assert isinstance(args.df, str)
    assert isinstance(args.nsplit, int) and args.nsplit >= 2
    assert isinstance(args.concat, list) and check_type_list(args.concat, list, int)
    for x in args.concat:
        for y in x:
            assert isinstance(y, int) and y >= 0 and y < args.nsplit
    assert isinstance(args.seed, int) and args.seed >= 0
    if args.splitby is not None:
        assert isinstance(args.splitby, list) and check_type_list(args.splitby, str)
    if args.ignan:
        assert args.splitby is not None
    df, ins_type = load_pickle(args.df)
    indexes      = np.arange(df.shape[0], dtype=int)
    if args.splitby is None:
        list_idxs = np.array_split(np.random.permutation(indexes), args.nsplit)
    else:
        for x in args.splitby: assert x in df.columns
        splitter   = MultilabelStratifiedKFold(n_splits=args.nsplit, shuffle=True, random_state=args.seed)
        ndf_target = None
        if ins_type == "polars":
            dfwk = df.select(args.splitby)
            if args.ignan:
                ndf_bool = dfwk.with_columns([
                    pl.col(pl.Float32).fill_nan(None).is_null(),
                    pl.col(pl.Float64).fill_nan(None).is_null(),
                    pl.col(pl.Int32).is_null(),
                    pl.col(pl.Int64).is_null(),
                ]).to_numpy().sum(axis=-1).astype(bool)
                indexes = indexes[~ndf_bool]
                dfwk    = dfwk.filter(~ndf_bool)
            ndf_target = dfwk.to_numpy()
        elif ins_type == "pandas":
            ndf_target = df[args.splitby].to_numpy()
            if args.ignan:
                ndf_bool   = np.isnan(ndf_target).sum(axis=-1).astype(bool)
                ndf_target = ndf_target[~ndf_bool]
                indexes    = indexes[~ndf_bool]            
        assert isinstance(ndf_target, np.ndarray)
        LOGGER.info(f"split by: {ndf_target}")
        if len(args.splitby) == 1:
            ndf_target = np.concatenate([ndf_target, np.zeros_like(ndf_target, dtype=int)], axis=-1)
        list_idxs = [index_valid for _, index_valid in splitter.split(indexes, ndf_target)]
    for idxs in args.concat:
        idx_new = np.concatenate([list_idxs[i] for i in idxs], axis=0)
        LOGGER.info(f"Concat: {idxs}. indexes: {idx_new}")
        if ins_type == "polars":
            df_new = df[idx_new]
            df_new.write_parquet(f"{args.df}.split{args.nsplit}.{'+'.join([str(i) for i in idxs])}.parquet", compression="zstd")
        elif ins_type == "pandas":
            df_new = df.iloc[idx_new]
            df_new.to_pickle(f"{args.df}.split{args.nsplit}.{'+'.join([str(i) for i in idxs])}.pickle")
        else:
            LOGGER.raise_error(f"Unknown dataframe type: {ins_type}", exception=ValueError(f"Unknown dataframe type: {ins_type}"))
        if list_df is not None:
            list_df.append(df_new)


if __name__ == "__main__":
    list_df = []
    split(args=args, list_df=list_df)
