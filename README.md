# Pandas - Cheatsheet

## Indexing dataframes

- Example 1:
```
index   col1    col2
3       1.0     2
100     2.2     1
'a'     3.1     2
```

note: the indexes of a dataframe can be arbitrary. so row 1 does not necesarily have index 0!

```python
df.iloc[0]     # first row
df.loc[0]      # key error
df.ix[0]       # first row
df.ix['a']     # third row
```

ix usually tries to behave like loc but falls back to behaving like iloc if a label is not present in the index.


- Example 2
```
     col1  col2
100     1     4
22      2     5
31      3     6
```

```python
df[df['col1'] == 1]
#      col1  col2
# 100     1     4
df.loc[df['col1'] == 1]
#     col1  col2
# 100     1     4
df.iloc[df['col1'] == 1]
# gives index error!!!
```

- reset_index (intruduces indices from 0 to len(df)-1)

```python
df.reset_index(drop=True)
# drop=True --> avoid the old index being added as a column:
```

## Dataframes v.s. Series
A Series can be understood as a single row of a DataFrame
- Example 1
```python
df = pd.DataFrame({'col1': [1,2,3], 'col2': [4,5,6]}, index=['a','b','c'])
sr = df.loc['b'] # get row with index b
# sr.name is 'b'
df = df.append(sr) # the name of the series will be used as row index in the dataframe

df.iloc[0] = sr # replaces first row of the dataframe with the series
```


## Append v.s. concatenate

- Append combines the rows of one dataframe to another dataframe
```python
df.append(other, ignore_index=False, verify_integrity=False, sort=None)
# ignore_index=True --> automatically resets indexes after appending
# ignore_index=False --> migth lead to duplicate indexes
# verify_integrity=True --> raise ValueError on creating index with duplicates.
```

- Concat is almost the same, but accepts list of more than two dataframes
Calling append with each dataframe will be less performant that using concat once on a list of dataframe since each concat and append call makes a full copy of the data.
```python
df.concat(df_list, axis=0, ignore_index=False, copy=True)
# copy=False --> do not copy data unnecessarily
```
ignore_index=True --> automatically resets indexes after concat

Example to build up a dataframe in a dynamic fashion (useful when size of dataframe is not known beforehand)
```python
# create an empty dataframe
df = pd.DataFrame(columns=['col1', 'col2'])

# or create an empty dataframe with same columns as another df
df = pd.DataFrame(columns=df_orig.columns.tolist())

for i in range(nr_iterations):
    df_newrows = ...
    df = pd.concat([df, df_newrows], axis=0)
    # df = pd.concat([df, df_newrows], axis=0, ignore_index=True)
    # df = df.append(df_newrows, ignore_index=True)
```


## Views v.s. copys / chained assignments

- modifying a view of a dataframe, modifies the original df
- modifying a copy does not alter the original df

- SettingWithCopyWarning
"A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead"

Pandas generates the warning when it detects something called chained assignment

"chaining": more than one indexing operation back-to-back; for example data[1:5][1:3]

e.g.:
```python
df[data.age == 21]['score'] = 100
df.loc[data['age'] == '21']['score'] = 100 # equivalent

# or in two lines
df_temp = df[df['age'] > 10]
df_temp['score'] = 99
```
triggers warning and results in failed assignment

reason: df[data.bidder == 'parakeet2004'] creates a copy of the dataframe!
 
### How to do it correctly?
make exactly a single call to one of the indexers:
```python
df.loc[df['age'] > 10, 'score'] = 99
```
condition > 10 applies to column 'age', while the assignment is done to column 'score'

```python
df[df.iloc[:, 12] == 2] = 99
```

Changes all rows of the 13rd column!!! note df[df.iloc[:, 'score'] == 2] would be invalid, you have to work with row/column numbers, and not row indexes/column names

Note: this now changes the values of df, BUT: the warning is still triggered (Pandas does not know if you want to modify the original DataFrame or just the first subset selection)
