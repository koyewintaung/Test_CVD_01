import pandas as pd
df = pd.read_csv("data/raw/heart.csv")   # or .xlsx
# normalize column names
df.columns = (df.columns.str.strip()
                          .str.lower()
                          .str.replace(' ', '_')
                          .str.replace('-', '_')
                          .str.replace('(', '')
                          .str.replace(')', ''))
df.info()
df.head()
df.describe(include='all').T
# target balance
df['target'].value_counts(normalize=True)
