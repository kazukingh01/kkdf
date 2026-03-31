# kkdf
my dataframe tools 

# Tests

```bash
python -m venv venv
source venv/bin/activate
pip install pytest
pip install -e .
python -m pytest tests/ -v
kkdfdiff --df1 tests/df1.pickle  --df2 tests/df2.pickle  --index id
kkdfdiff --df1 tests/df1.parquet --df2 tests/df2.parquet --index id
kkdfvar  --df tests/df1.pickle
kkdfconcat tests/df1.parquet tests/df2.parquet --output /tmp/out.parquet
kkdfsplit --df tests/df1.pickle --nsplit 3 --concat 0+1,2
kkdfconcat tests/df_col1.parquet tests/df_col2.parquet --output /tmp/out_col.parquet # error is fine
kkdfconcat tests/df_col1.parquet tests/df_col2.parquet --output /tmp/out_col.parquet --ignore-cols
```
