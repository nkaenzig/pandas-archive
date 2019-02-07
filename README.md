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
Make exactly a single call to one of the indexers:
```python
df.loc[df['age'] > 10, 'score'] = 99
```
Condition > 10 applies to column 'age', while the assignment is done to column 'score'

```python
df[df.iloc[:, 12] == 2] = 99
```

Changes all rows of the 13rd column!!! note df[df.iloc[:, 'score'] == 2] would be invalid, you have to work with row/column numbers, and not row indexes/column names

Note: this now changes the values of df, BUT: the warning is still triggered (Pandas does not know if you want to modify the original DataFrame or just the first subset selection)




# Working with big datasets
## Reading in chunks
```python
df = pd.read_csv(r'../input/data.csv', chunksize=1000000)

chunks_list = []  # append each chunk df here 

# Each chunk is in df format
for df_chunk in df:  
    # perform data processing and filtering 
    df_chunk_processed = chunk_preprocessing(df_chunk)
    
    # Once the data filtering is done, append the chunk to list
    chunks_list.append(chunk_filter)
    
# concat the list into one dataframe 
df_concat = pd.concat(chunks_list)
```

## dtypes
When a dataset gets larger, we need to convert the dtypes in order to save memory. 
By default integer types are int64 and float types are float64, REGARDLESS of platform (32-bit or 64-bit). The following will all result in int64 dtypes.

- Example 1
```python
df[int_columns] = df[int_columns].astype('int8')
df[int_columns] = df[int_columns].astype('int16')
df[int_columns] = df[int_columns].astype('int32')
df[float_columns] = df[float_columns].astype('float8')
df[float_columns] = df[float_columns].astype('float16')
df[float_columns] = df[float_columns].astype('float32')
```

# Dataframe filtering and processing

## Dealing with missing values (np.nan)
dropna() (which removes NA values) and fillna()

Often the missing values are not np.nan but have another value like 0 or '-'.
You have to replace these first with np.nan to use pandas N/A-functions.
```python
df.replace({1: np.nan})
df.replace({'-': np.nan})
# or one-liners:
df.replace(to_replace={0: np.nan, '-': np.nan})
df.replace(to_replace=[0, '-'], value=np.nan)

# caution: this, might change the dtypes of the df columns
```

Now we can drop rows/cols with missing values with dropna()
Note: use how or thresh parameters:
```
how='any': at least one N/A (default)
how='all': all entries have to be N/A
thresh=3: minimum number of non-null values for the row/column to be kept:
```

For columns, where only a few values are missing, it might be meaningful to replace the NaN values with the mean value (for numerical columns) or the most frequent value (for categorical columns):

```python
# mean
df = df.fillna(df.mean())
# most frequent value
df = df.fillna(df.mode())
# outdated method for most frequent value (interesting for learning)
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
```

```python
df.dropna(axis=0) # drops all ROWS with at least one N/A entry
df.dropna(axis=0, how='all') # drops all ROWS where all elements are N/A
df.dropna(axis=1) # drops all COLUMNS with at least one N/A entry

# drop all columns where more than 90% of the values are N/A
df.dropna(axis=1, thresh=int(0.9*df.shape[0]))

```

## Categorical columns
Split dataframe into categorical and numerical columns
```python
df_cat = df.select_dtypes(include=['object'])
df.drop(df_cat.columns, axis=1, inplace=True) 
# df now holds only numerical columns
```

See helper functions in pandas-helpers.py (""" FUNCTIONS FOR CATEGORICAL COLUMNS """)

## Replace/remove strings in df columns
```python

df[col].replace('\n', '', regex=True, inplace=True)

# remove all the characters after &# (including &#) for column - col_1
df[col].replace(' &#.*', '', regex=True, inplace=True)

# remove white space at the beginning of string 
df[col] = df[col].str.lstrip()
```

# Preprocessing
## Standardization
```python
# good way
df = (df-df.mean())/df.std()
# bad way
df = df.apply(lambda x:(x-x.mean())/x.std())
```

# Plotting
## Frequency Plot of a columns
```python
df[column_name].value_counts().plot.bar()
```

## Box Plots
Generate boxplots of one ore multiple columns
```python
df_num[['LotFrontage', 'OverallQual', 'MasVnrArea']].plot.box()
```