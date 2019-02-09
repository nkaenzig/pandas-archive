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
df_num = df.select_dtypes(exclude=['object'])
# or delete categorical colums from df
df.drop(df_cat.columns, axis=1, inplace=True) 
# or
df_num = df._get_numeric_data()
```

Enumerate categories
```python
df_cat.astype('category')
# A
df_cat = df_cat.apply(lambda x: x.cat.codes)
# B
for col_name in df_cat:
    df_cat[col_name] = df_cat[col_name].cat.codes
```

One-hot encoding
```python
for col_name in df_cat:
     df_dummies = pd.get_dummies(df[col_name], prefix='category')
     df_cat = pd.concat([df_cat, df_dummies], axis=1)
     df_cat = df.drop(col_name, axis=1)
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
# using sklearn
from sklearn.preprocessing import StandardScaler
df = pd.DataFrame(scaler.fit_transform(df.values))
```

# Reshaping
## Grouping

## Pivoting
[Pivoting](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping)
- example
```python
import seaborn as sns
df_flights_long = sns.load_dataset("flights")
df_flights = flights_long.pivot(index="month", columns="year", values="passengers")

# df_flights_long
year	month	passengers
0	1949	January	112
1	1949	February	118
2	1949	March	132
.
.
.

# df_flights
year	1949	1950	1951	1952	1953	1954	1955	1956	1957	1958	1959	1960
month												
January	112	115	145	171	196	204	242	284	315	340	360	417
February	118	126	150	180	196	188	233	277	301	318	342	391
March	132	141	178	193	236	235	267	317	356	362	406	419
```

# MultiIndexing
You can think of MultiIndex as an array of tuples where each tuple is unique. A MultiIndex can be created from a list of arrays (using MultiIndex.from_arrays()), an array of tuples (using MultiIndex.from_tuples()), a crossed set of iterables (using MultiIndex.from_product()), or a DataFrame (using MultiIndex.from_frame()).

"Multi-Columns"
```python
In [84]: cols = pd.MultiIndex.from_tuples([(x, y) for x in ['A', 'B', 'C']
   ....:                                   for y in ['O', 'I']])
   ....: 

In [85]: df = pd.DataFrame(np.random.randn(2, 6), index=['n', 'm'], columns=cols)

In [86]: df
Out[86]: 
          A                   B                   C          
          O         I         O         I         O         I
n  1.920906 -0.388231 -2.314394  0.665508  0.402562  0.399555
m -1.765956  0.850423  0.388054  0.992312  0.744086 -0.739776
```

"Multi-Index"
```python
In [8]: iterables = [['bar', 'baz', 'foo', 'qux'], ['one', 'two']]

In [9]: index = pd.MultiIndex.from_product(iterables, names=['first', 'second'])
In[10]: index 
Out[10]: 
MultiIndex(levels=[['bar', 'baz', 'foo', 'qux'], ['one', 'two']],
           codes=[[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1]],
           names=['first', 'second'])

In [11]: pd.Series(np.random.randn(8), index=index)
Out[11]: 
first  second
bar    one       0.469112
       two      -0.282863
baz    one      -1.509059
       two      -1.135632
foo    one       1.212112
       two      -0.173215
qux    one       0.119209
       two      -1.044236
dtype: float64
```

## caution with chained indexing
```python
Out[340]: 
    one          two       
  first second first second
0     a      b     c      d
1     e      f     g      h
2     i      j     k      l
3     m      n     o      p

# chained --> value assignments will fail
dfmi['one']['second']

# correct: single access
dfmi.loc[:, ('one', 'second')]
```

# Configure Pandas
```python
import pandas as pd

def pd_config():
    options = {
        'display': {
            'max_colwidth': 25,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 14,
            'max_seq_items': 50,         # Max length of printed sequence
            'precision': 4,
            'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+

if __name__ == '__main__':
    pd_config()
```

# Datetime 
Concept | Scalar Class | Array Class | pandas Data Type | Primary Creation Method
--- | --- | --- | --- | ---
Date times | Timestamp | DatetimeIndex | datetime64[ns] or datetime64[ns, tz] | to_datetime or date_range
Time deltas | Timedelta | TimedeltaIndex | timedelta64[ns] | to_timedelta or timedelta_range
Time spans | Period | PeriodIndex | period[freq] | Period or period_range
Date offsets | DateOffset | None | None | DateOffset

[Pandas Timeseries Reference](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#)
[Pandas Time/Date Components](https://pandas.pydata.org/pandas-docs/stable/user_guide/)timeseries.html#time-date-components
[Pandas Frequency Strings](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects)

## Timezones
- To supply the time zone, you can use the tz keyword to date_range and other functions. 
```python
# list of all available timezones
from pytz import common_timezones, all_timezones
# some common values: 'UTC', 'GMT', 'US/Central', 'US/Eastern', 'Europe/Rome'
# UTC and GMT are same practically, but GMT is a time zone and UTC is a time standard.
```
- Under the hood, all timestamps are stored in UTC --> comparing Timestamps with with different timezones but same UTC time yields True
```python
rng_utc = pd.date_range('3/6/2012 00:00', periods=5, freq='D', tz='UTC')
rng_eastern = rng_utc.tz_convert('US/Eastern')
rng_berlin = rng_utc.tz_convert('Europe/Berlin')
rng_eastern == rng_berlin

Out[0]: pandas._libs.tslibs.timestamps.Timestamp
```

- Create pandas.core.indexes.datetimes.DatetimeIndex series
```python
pd.date_range(start='1/1/2018', periods=12, freq='M')
pd.date_range(start='2018-04-24', end='2018-04-27', freq='D')
pd.date_range(start='2018-04-24', freq='Y', periods=10)
pd.date_range(start='2018-04-24', end='2018-04-27', periods=1000)
```

## Convert non datetime date to datetime
```python
# create daterange (freq: {'D', 'M', 'Q', 'Y'})
In [5]: sr_dates = pd.Series(pd.date_range('2017', periods=4, freq='Q'))

In [6]: sr_dates
Out[6]: 
0   2017-03-31
1   2017-06-30
2   2017-09-30
3   2017-12-31
dtype: datetime64[ns]

# get weekdays of elements in datetime series
In [7]: sr_dates.dt.day_name()
Out[7]: 
0      Friday
1      Friday
2    Saturday
3      Sunday
dtype: object

# convert a single datecolumn to datetime series
In [20]: df = pd.DataFrame(np.ones(5), columns=['Date'])

In [21]: df['Date'] = '2017-12##13'

In [22]: df
Out[22]: 
          Date
0  2017-12##13
1  2017-12##13
2  2017-12##13
3  2017-12##13
4  2017-12##13

In [23]: df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m##%d')
Out[23]: 
0   2017-12-13
1   2017-12-13
2   2017-12-13
3   2017-12-13
4   2017-12-13
Name: Date, dtype: datetime64[ns]

# Create a DatetimeIndex From Component Columns
>>> from itertools import product
>>> datecols = ['year', 'month', 'day']

date_tuples = list(product([2017, 2016], [1, 2], [1, 2, 3])) # list with tuples (year, month, day)
>>> df = pd.DataFrame(date_tuples, columns=datecols)
>>> df['data'] = np.random.randn(len(df))
>>> df
    year  month  day    data
0   2017      1    1 -0.0767
1   2017      1    2 -1.2798
2   2017      1    3  0.4032
3   2017      2    1  1.2377
4   2017      2    2 -0.2060
5   2017      2    3  0.6187
6   2016      1    1  2.3786
7   2016      1    2 -0.4730
8   2016      1    3 -2.1505
9   2016      2    1 -0.6340
10  2016      2    2  0.7964
11  2016      2    3  0.0005

# the date column names must be named: [‘year’, ‘month’, ‘day’, ‘minute’, ‘second’, ‘ms’, ‘us’, ‘ns’]) 
>>> df.index = pd.to_datetime(df[datecols])
>>> df.head()
            year  month  day    data
2017-01-01  2017      1    1 -0.0767
2017-01-02  2017      1    2 -1.2798
2017-01-03  2017      1    3  0.4032
2017-02-01  2017      2    1  1.2377
2017-02-02  2017      2    2 -0.2060
```
Note: If you pass a DataFrame to to_datetime() as argument, it will look for columns with names ‘year’, ‘month’, ‘day’, etc. and will set the indexx of the dataframe to the datetimes it extracted from these columns.
If you pass a Series to to_datetime() holding raw string dates, you must specify the format of these strings with format=..., and pandas will then return a Series with dtype=datetime. 
```
format example:
'05SEP2014:00:00:00.000'
'%d%b%Y:%H:%M:%S.%f'
```

# MISC
## Create a dictionary from two DataFrame Columns
```python
dictionary = pd.Series(df['val_col'].values, index=df['key_col']).to_dict()
```

## isin()
```python
df[~df[col_name].isin(a_set)]
df[~df[col_name].isin(a_list)]
df[~df[col_name].isin(a_dict)]
df[~df.index.isin(a_list)]
```

## Creating toy DataFrames
Useful for testing, exploring new pandas methods.
Note: 
```python
# standard index
df = pd.DataFrame(np.random.randn(1000, 4), columns=list('ABCD'))

# timeseries index
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))
```

## Accessor methods
```python
>>> pd.Series._accessors
{'cat', 'str', 'dt'}
```

### String accessor .str (for dtype='object')
```python
# convert column to UPPERCASE
df[col_name].str.upper()

# count string occurence in each column
df[col_name].str.count(r'\d') # counts number of digits

```

Split string column in multiple columns using extract(regex)
```python
In [2]: addr = pd.Series([
   ...:      'Washington, D.C. 20003',
   ...:      'Brooklyn, NY 11211-1755',
   ...:      'Omaha, NE 68154',
   ...:      'Pittsburgh, PA 15211'
   ...:  ])

In [3]: regex = (r'(?P<city>[A-Za-z ]+), '       # One or more letters followed by ,
   ...:           r'(?P<state>[A-Z]{2}) '        # 2 capital letters
   ...:           r'(?P<zip>\d{5}(?:-\d{4})?)')  # 5-digits + optional 4-digit extension

In [4]: addr.str.replace('.', '').str.extract(regex)
Out[4]: 
         city state         zip
0  Washington    DC       20003
1    Brooklyn    NY  11211-1755
2       Omaha    NE       68154
3  Pittsburgh    PA       15211
```

## Get memory usage
```python
# get memory usage of each column
df.memory_usage(index=False, deep=True)

# compare to memory usage after type conversion
df.astype('category').memory_usage(index=False, deep=True)
```

# Resources
- [Pandas Cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#cookbook-selection)
This is a repository for short and sweet examples and links for useful pandas recipes. 

