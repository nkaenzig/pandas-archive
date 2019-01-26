# Pandas - Cheatsheet

## Indexing dataframes

- example:
index   col1    col2
3       1.0     2
100     2.2     1
'a'     3.1     2

note: the indexes of a dataframe can be arbitrary. so row 1 does not necesarily have index 0!

>>> df.iloc[0] --> first row
>>> df.loc[0] --> key error
>>> df.ix[0] --> first row
>>> df.ix['a']  --> third row

ix usually tries to behave like loc but falls back to behaving like iloc if a label is not present in the index.


- example 2
     col1  col2
100     1     4
22      2     5
31      3     6

>>> df[df['col1'] == 1]
     col1  col2
100     1     4
>>> df.loc[df['col1'] == 1]
     col1  col2
100     1     4
>>> df.iloc[df['col1'] == 1]
--> gives index error!!!

- reset index (intruduces indices from 0 to len(df)-1)
Use the drop parameter to avoid the old index being added as a column:

< df.reset_index(drop=True)

## Append v.s. concatenate

- Append combines the rows of one dataframe to another dataframe
```
df.append(other, ignore_index=False, verify_integrity=False, sort=None)
```
ignore_index=True --> automatically resets indexes after appending
ignore_index=False --> migth lead to duplicate indexes
verify_integrity=True --> raise ValueError on creating index with duplicates.


- Concat is almost the same, but accepts list of more than two dataframes
calling append with each dataframe will be less performant that using concat once on a list of dataframe since each concat and append call makes a full copy of the data.
```
df.concat
```
ignore_index=True --> automatically resets indexes after concat

Example to build up a dataframe in a dynamic fashion (useful when size of dataframe is not known beforehand)
```
# create an empty dataframe
df = pd.DataFrame(columns=['col1', 'col2'])

# create an empty dataframe with same columns as another df
df = pd.DataFrame(columns=df_orig.columns.tolist())

for i in range(nr_iterations):
    df_newrows = ...
    df = pd.concat([df, df_newrows], axis=0)
    # df = pd.concat([df, df_newrows], axis=0, ignore_index=True)
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
```
df[data.bidder == 'parakeet2004']['bidderrate'] = 100
df.loc[data['bidder'] == 'parakeet2004']['bidderrate'] = 100
```

or in two lines:
```
df_temp = df[df['age'] > 10]
df_temp['score'] = 99
```
--> triggers warning
--> failed assignment, as 

reason: df[data.bidder == 'parakeet2004'] creates a copy of the dataframe!
 
HOW TO DO IT CORRECTLY?
make exactly a single call to one of the indexers:
```
df.loc[df['age'] > 10, 'score'] = 99
```
condition > 10 applies to column 'age', while the assignment is done to column 'score'

```
df[df.iloc[:, 12] == 2] = 99
```

--> changes all rows of the 13rd column!!! note df[df.iloc[:, 'score'] == 2] would be invalid, you have to work with row/column numbers, and not row indexes/column names

note: this now changes the values of df, BUT: the warning is still triggered (Pandas does not know if you want to modify the original DataFrame or just the first subset selection)

  """
