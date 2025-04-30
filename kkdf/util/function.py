import pickle
import pandas as pd
import polars as pl
import numpy as np
from kklogger import set_logger
from kkdf.util.com import check_type_list
LOGGER = set_logger(__name__)


__all__ = [
    "load_pickle",
    "check_pandas_diff",
    "check_polars_diff",
]


def load_pickle(path: str) -> tuple[pl.DataFrame | pd.DataFrame | object, str]:
    assert isinstance(path, str)
    ins_type = None
    try:
        df = pd.read_pickle(path)
        ins_type = "pandas"
    except Exception as e:
        # Not pandas
        try:
            df = pl.read_parquet(path)
            ins_type = "polars"
        except pl.exceptions.ComputeError as e:
            # Not polars with parquet
            try:
                df = pl.read_ipc(path)
                ins_type = "polars"
            except pl.exceptions.ComputeError as e:
                # Not polars with IPC
                try:
                    with open(path, "rb") as f:
                        df = pickle.load(f)
                except pl.exceptions.ComputeError as e:
                    LOGGER.raise_error(f"It might be that polars version ({pl.__version__})is different when you saved the dataframe.", exception=e)
    if isinstance(df, pl.DataFrame):
        ins_type = "polars"
    elif isinstance(df, pd.DataFrame):
        ins_type = "pandas"
    else:
        LOGGER.raise_error(f"Unknown dataframe type: {type(df)}", exception=ValueError(f"Unknown dataframe type: {type(df)}"))
    return df, ins_type

def check_index_pandas(ndf1, ndf2, text: str="index"):
    if len(ndf1) == len(ndf2) == ndf1.isin(ndf2).sum() == ndf2.isin(ndf1).sum():
        LOGGER.info(f"all {text} is SAME", color=["BOLD", "BLUE"])
        return ndf1
    else:
        LOGGER.warning(f"{text} is different.")
        ndf_same = ndf1[ndf1.isin(ndf2)].values
        if len(ndf_same) == 0:
            LOGGER.warning(f"All {text} is different.")
            raise
        LOGGER.info(f"same {text}: {ndf_same}")
        LOGGER.warning(f"only df1 {text}: {ndf1[~ndf1.isin(ndf2)].values}")
        LOGGER.warning(f"only df2 {text}: {ndf2[~ndf2.isin(ndf1)].values}")
        return ndf_same

def check_pandas_diff(df1: pd.DataFrame, df2: pd.DataFrame, value_fillnan: object=-9999):
    assert isinstance(df1, pd.DataFrame)
    assert isinstance(df2, pd.DataFrame)
    assert isinstance(value_fillnan, (int, float, str, bool, None))
    df1, df2 = df1.copy().fillna(value_fillnan), df2.copy().fillna(value_fillnan)
    LOGGER.info("check dataframe shape.", color=["BOLD", "GREEN"])
    LOGGER.info(f"df1 shape: {df1.shape}")
    LOGGER.info(f"df2 shape: {df2.shape}")
    LOGGER.info("check dataframe index.", color=["BOLD", "GREEN"])
    # check row index
    index1, index2 = df1.index, df2.index
    index_same = check_index_pandas(index1, index2, text="index")
    # check col index
    columns1, columns2 = df1.columns, df2.columns
    columns_same = check_index_pandas(columns1, columns2, text="columns")
    if len(df1.index) != len(df1.index.unique()): LOGGER.raise_error(f"df1 index is not unique")
    if len(df2.index) != len(df2.index.unique()): LOGGER.raise_error(f"df2 index is not unique")
    if len(df1.columns) != len(df1.columns.unique()): LOGGER.raise_error(f"df1 columns is not unique")
    if len(df2.columns) != len(df2.columns.unique()): LOGGER.raise_error(f"df2 columns is not unique")
    LOGGER.info("we check only same indexes and same columns", color=["BOLD", "GREEN"])
    df1 = df1.loc[index_same, columns_same]
    df2 = df2.loc[index_same, columns_same]
    LOGGER.info("check whole data.", color=["BOLD", "GREEN"])
    sebool = (df1 == df2).sum() == df1.shape[0]
    for x in sebool.index[sebool.values]:
        LOGGER.info(f"same column: {x}", color=["BOLD", "BLUE"])
    for x in sebool.index[~sebool.values]:
        LOGGER.warning(f"diff column: {x}")
        index = (df1[x] != df2[x])
        LOGGER.info(f"df1: \n{df1.loc[index, x]}")
        LOGGER.info(f"df2: \n{df2.loc[index, x]}")
    return df1, df2

def check_polars_diff(df1: pl.DataFrame, df2: pl.DataFrame, indexes: list[str]=None):
    assert isinstance(df1, pl.DataFrame)
    assert isinstance(df2, pl.DataFrame)
    assert isinstance(indexes, (list, tuple))
    assert check_type_list(indexes, str)
    for x in indexes:
        assert x in df1.columns
        assert x in df2.columns
    df1, df2 = df1.clone(), df2.clone()
    LOGGER.info("check dataframe shape.", color=["BOLD", "GREEN"])
    LOGGER.info(f"df1 shape: {df1.shape}")
    LOGGER.info(f"df2 shape: {df2.shape}")
    LOGGER.info("check dataframe index.", color=["BOLD", "GREEN"])
    # check row index
    assert np.all(df1[indexes].with_columns(pl.all().unique().len() == pl.len())[0].to_numpy().reshape(-1)) # Index must be unique
    assert np.all(df2[indexes].with_columns(pl.all().unique().len() == pl.len())[0].to_numpy().reshape(-1)) # Index must be unique
    df1        = df1.sort(by=indexes, descending=False)
    df2        = df2.sort(by=indexes, descending=False)
    ndf_idx1   = df1[indexes].to_pandas().set_index(indexes).index.copy()
    ndf_idx2   = df2[indexes].to_pandas().set_index(indexes).index.copy()
    index_same = check_index_pandas(ndf_idx1, ndf_idx2, text="index")
    df1        = df1.filter(ndf_idx1.isin(index_same))
    df2        = df2.filter(ndf_idx2.isin(index_same))
    # check col index
    index_same = check_index_pandas(pd.Index(df1.columns), pd.Index(df2.columns), text="columns")
    df1 = df1[:, index_same]
    df2 = df2[:, index_same]
    LOGGER.info("we check only same indexes and same columns", color=["BOLD", "GREEN"])
    LOGGER.info("check whole data.", color=["BOLD", "GREEN"])
    dfbool = df1.with_columns([
        ((pl.col(x) == df2[x]) | (pl.col(x).is_null() & df2[x].is_null())).alias(x)
        for x in df1.columns
    ])
    dfbool  = dfbool.with_columns(pl.all().fill_null(False))
    ndfbool = np.array([x.all() for x in dfbool], dtype=bool)
    dfbool  = dfbool.select(
        np.array(dfbool.columns, dtype=object)[ ndfbool].tolist() + 
        np.array(dfbool.columns, dtype=object)[~ndfbool].tolist()
    )
    ndfbool = ((dfbool.to_numpy() == False).sum(axis=-1) > 0)
    LOGGER.info(   f"all same data indexes: {df1[indexes].filter(~ndfbool)}", color=["BOLD", "BLUE"])
    LOGGER.warning(f"some different data indexes: {df1[indexes].filter( ndfbool)}")
    ndf_index = df1[indexes].to_numpy()
    for sewk in dfbool:
        if sewk.all():
            LOGGER.info(f"same column: {sewk.name}", color=["BOLD", "BLUE"])
        else:
            LOGGER.warning(f"diff column: {sewk.name}")
            LOGGER.info(f"idx: \n{ndf_index[~(sewk.to_numpy())]}") 
            LOGGER.info(f"df1: \n{df1.filter(~sewk)[sewk.name]}")
            LOGGER.info(f"df2: \n{df2.filter(~sewk)[sewk.name]}")
    return df1, df2

def get_variance(df: pl.DataFrame | pd.DataFrame, check_ratio: float=0.95, n_divide: int=10000, n_display: int=5):
    assert isinstance(df, (pl.DataFrame, pd.DataFrame))
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    assert isinstance(check_ratio, float) and 0.0 < check_ratio < 1.0
    assert isinstance(n_divide, int) and n_divide >= 1000
    assert isinstance(n_display, int) and n_display >= 1
    # check Nan
    dfnan = df.select([pl.col(pl.Float32).is_nan(), pl.col(pl.Float64).is_nan()]).sum()
    for sewk in dfnan:
        if sewk[0] > 0:
            LOGGER.warning(f"Nan values, {sewk.name}: {sewk[0]}")
    df = df.with_columns([
        pl.col(pl.Float32).fill_nan(None),
        pl.col(pl.Float64).fill_nan(None),
    ]).with_columns(pl.all().sort())
    n_data = df.shape[0]
    for ignore_nan in [True, False]:
        LOGGER.info(f"Case of ignore_nan: {ignore_nan}")
        se_null = df.null_count()
        if ignore_nan:
            listwk  = []
            for sewk in (se_null == n_data):
                if sewk[0]:
                    LOGGER.warning(f"{sewk.name} has all NaN values.")
                else:
                    listwk.append(sewk.name)
            se_null = se_null.select(listwk)
            dictwk  = {x.name: np.round(np.linspace(x[0], n_data - 1, n_divide)).astype(int) for x in se_null}
            dfwk    = pl.concat([df[x][y].to_frame() for x, y in dictwk.items()], how="horizontal")
        else:
            idx  = np.round(np.linspace(0, n_data - 1, n_divide)).astype(int)
            dfwk = df[idx].clone()
        idx    = np.arange(n_divide, dtype=int)
        idx    = np.stack([idx, idx + int(n_divide * check_ratio)]).T
        idx    = idx[np.sum(idx >= n_divide, axis=1) == 0]
        dfwk1  = dfwk[idx[:, 0]]
        dfwk2  = dfwk[idx[:, 1]]
        sebool = (dfwk1.with_columns([(pl.col(x) == dfwk2[x]) | (pl.col(x).is_null() & dfwk2[x].is_null()) for x in dfwk1.columns]).sum() > 0)
        for sewk in sebool:
            if sewk[0]:
                for x, y in df[sewk.name].value_counts().sort("count", descending=True)[:n_display].iter_rows():
                    LOGGER.warning(f"[{int(ignore_nan)}] {sewk.name}: {x} count {y}, null {se_null[sewk.name][0]}")
