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

- Example 2
```python
df[['floor']] # returns a dataframe with one column
df['floor'] # returns a series (much faster)
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
## where & masking
```python
# | where: Replace values by NaN where condition is False
df.where(df>3)
df['col'].where(df>3)

# | mask: Replace values by NaN where condition is True
df.mask(df>3)
df['col'].mask(df>3)

# | replace other values by 2
df.where(df>2, other=2)
df.mask(df>2, other=2)

# | negate other values
df.where(df>2, other=-df)
df.mask(df>2, other=-df)

# | apply funciton to other values
df.where(df>2, lambda x: x*4)
df.mask(df>2, lambda x: x*4)

# or
def foo(x):
    return x*4

df.where(df>2, foo)
df.mask(df>2, foo)
```

## Split dataframe into features & labels
```python
df_labels = df.pop(label_colname)
df_features = df
```

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
## Group By (split-apply-combine)
1. <strong>Splitting</strong> the data into groups based on some criteria.
2. <strong>Applying</strong> a function to each group independently.
3. <strong>Combining</strong> the results into a data structure.

### "Splitting" / Grouping
```python
grouped = df.groupby('Gender')
grouped = df.groupby(['Gender', 'Age']) # this creates groups of the same gender & same age

# | If the axis is a MultiIndex (hierarchical), group by a particular level or levels
Heart Disease          Yes     No   
High Blood Pressure    Yes No Yes No
Sex    Marital Status              
Female Single            5  0   3  3
       Married           7  9   3  5
Male   Single            2  4   7  6
       Married           8  8   1  6

df.groupby(level=0).sum() # groups by first index (Sex), and calculates sum of Single and Married
df.groupby(level='Sex').sum() # works only if indices have names defined

Heart Disease	    Yes	    No
High Blood Pressure	Yes	No	Yes	No
Sex				
Female	            5	10	3	4
Male	            8	14	13	6

# | using function to create groups
def get_letter_type(letter):
    if letter.lower() in 'aeiou':
        return 'vowel'
    else:
        return 'consonant'

grouped = df.groupby(get_letter_type, axis=1) # axis=1: split by columns and not by index

# | By default the group keys are sorted during the groupby operation. You may however pass sort=False for potential 
df2.groupby(['X'], sort=False).sum()

# | access a group
df3.groupby(['Gender']).get_group('Female')
# Or for an object grouped on multiple columns:
df.groupby(['Gender', 'Age']).get_group(('Male', 26))
```
- Iterating through groups
df.groupby(...) returns a GroupBy object (a DataFrameGroupBy or SeriesGroupBy), and with this, you can iterate through the groups

```python
grouped = df.groupby('A')

for name, group in grouped:
    # | group is a dataframe
    ...
```

- GroupBy object attributes
```python
In [27]: df.groupby('A').groups
Out[27]: 
{'bar': Int64Index([1, 3, 5], dtype='int64'),
 'foo': Int64Index([0, 2, 4, 6, 7], dtype='int64')}

 In [34]: gb.<TAB>  # noqa: E225, E999
gb.agg        gb.boxplot    gb.cummin     gb.describe   gb.filter     gb.get_group  gb.height     gb.last       gb.median     gb.ngroups    gb.plot       gb.rank       gb.std        gb.transform
gb.aggregate  gb.count      gb.cumprod    gb.dtype      gb.first      gb.groups     gb.hist       gb.max        gb.min        gb.nth        gb.prod       gb.resample   gb.sum        gb.var
gb.apply      gb.cummax     gb.cumsum     gb.fillna     gb.gender     gb.head       gb.indices    gb.mean       gb.name       gb.ohlc       gb.quantile   gb.size       gb.tail       gb.weight
```

TODO: https://stackoverflow.com/questions/44635626/rename-result-columns-from-pandas-aggregation-futurewarning-using-a-dict-with


### "Applying" (Aggregation, Transformation, Filtration)
- (Std & Custom Aggregations)
Aggregation via the aggregate() or equivalently agg() method:
The result of the aggregation will have the group names as the new index along the grouped axis. In the case of multiple keys, the result is a MultiIndex by default, though this can be changed by using the as_index option.

[List of standard aggregation functions](https://pandas-docs.github.io/pandas-docs-travis/user_guide/groupby.html#aggregation)

```python
# | use pandas standard aggregation methods
df.groupby('user_id')['purchase_amount'].sum()
# | is the same as
df.groupby('user_id')['purchase_amount'].agg('sum')
# | is the same as
df.groupby('user_id')['purchase_amount'].agg(np.sum)

# multiple aggregations at once
df.groupby('A')['C'].agg([np.sum, np.mean, np.std])

          sum      mean       std
A                                
bar  0.392940  0.130980  0.181231
foo -1.796421 -0.359284  0.912265

# | apply sum to all numerical columns
df.groupby('A').agg('sum')
df.groupby('A').agg([np.sum, np.mean, np.std])

            C                             D                    
          sum      mean       std       sum      mean       std
A                                                              
bar  0.392940  0.130980  0.181231  1.732707  0.577569  1.366330
foo -1.796421 -0.359284  0.912265  2.824590  0.564918  0.884785



# | don't use group names as new index
df.groupby(['A', 'B'], as_index=False).agg(np.sum)

# | get size of each group
df.groupby('Gender').size()

# | get group stats
df.groupby('Gender').describe()
```

```python
# | custom aggregation
df.groupby('Gender').agg({'C': np.sum, 'Age': lambda x: np.std(x, ddof=1)})


def concat_prod(sr_products):
    # | sort product descriptions by length (#characters)
    sr_products = sr_products.astype(str)
    new_index = sr_products.str.len().sort_values(ascending=False).index
    sr_products = sr_products.reindex(new_index)
    # | aggregation
    if len(sr_products) > 2:
        return '~~~'.join(sr_products)
    else:
        return ''.join(sr_products)

# | Method A
# | Problem, in df_new, only the columns match_columns & lic_prod_cn will be included
df_new = df_lic.groupby(match_columns)[lic_prod_cn].agg(concat_prod).reset_index()

# | Method B
# | if you want to keep all columns (e.g. just keep the first value of the corresponding groups, for columns that are not aggregated)
agg_dict = {col_name: 'first' for col_name in df_lic if col_name not in match_columns}
agg_dict[lic_prod_cn] = concat_prod

df_lic = df_lic.groupby(match_columns).agg(agg_dict).reset_index()
```

```python
# | How can I “merge” rows by same value in a column in Pandas with aggregation functions?
aggregation_functions = {'price': 'sum', 'amount': 'sum', 'name': 'first'}
df_new = df.groupby(df['id']).aggregate(aggregation_functions)
```
Note: agg is an alias for aggregate. Use the alias.

Pandas custom aggregators:
```
date, cardio_time, muscles, muscle_time, stretch_time
2018-01-01, 0, "biceps / lats", 40, 5
2018-01-02, 30, "", 0, 10
2018-01-03, 0, "lats / calf", 41, 6
2018-01-03, 30, "hamstring", 4, 5
2018-01-04, 0, "biceps / lats", 42, 8

TO

2018-01-01, 0, "biceps / lats", 40, 5
2018-01-02, 30, "", 0, 10
2018-01-03, 30, "lats / calf / hamstring", 45, 11
2018-01-04, 0, "biceps / lats", 42, 8
```
```python
custom_aggregator = lambda a: " / ".join(a) 
data_.groupby(by='date').agg({'muscle_time': 'sum',
                              'stretch_time': 'sum',
                              'cardio_time': 'sum',
                              'muscles': custom_aggregator}).reset_index()
```

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
Both the .index as well as the .columns of a DataFrame can have various levels.
In the following example, the index has two levels: ['Sex', 'Marital Status'], and the columns have two levels:['Heart Disease', 'High Blood Pressure'].

```python
colidx = pd.MultiIndex.from_product([('Yes', 'No'), ('Yes', 'No')],
                                    names=['Heart Disease', 'High Blood Pressure'])
rowidx = pd.MultiIndex.from_product([('Female', 'Male'), ('Single', 'Married')], 
                                    names=['Sex', 'Marital Status'])

df = pd.DataFrame(np.random.randint(10, size=(4, 4)), index=rowidx, columns=colidx)


Heart Disease          Yes     No   
High Blood Pressure    Yes No Yes No
Sex    Marital Status              
Female Single            5  0   3  3
       Married           7  9   3  5
Male   Single            2  4   7  6
       Married           8  8   1  6

# | reset_index(): Reset the index of the DataFrame, and use the default one instead. If the DataFrame has a MultiIndex, this method can remove one or more levels
df.reset_index(level=0) # this makes a column out of the index column "Sex".
df.reset_index() # this converts all indices to regular columns - introduces a RangeIndex

# | select columns in multiindex columns
df.iloc[:, df.columns.get_level_values(1)=='Yes']

Heart Disease	    Yes	No
High Blood Pressure	Yes	Yes
Sex	    Marital Status		
Female	Single	    5	5
        Married	    5	4
Male	Single	    4	9
        Married	    2	0

# | swap the levels 0&1
df.swaplevel(0, 1, axis=0)

# | drop a level of index or column
df.index = df.index.droplevel(level=0)
df.columns = df.columns.droplevel(level=0)

# | set index
df = df.reset_index('Marital Status') # make column out of index
df = df.set_index('Marital Status') # make index out of column again

df = df.set_index(['A', 'B'])

# | accessing
df3.loc[('Female', 'Single'), :]

Heart Disease  High Blood Pressure
Yes            Yes                    0
               No                     4
No             Yes                    5
               No                     5
Name: (Female, Single), dtype: int32

df.loc[('Female', 'Single'), 'Yes']

High Blood Pressure
Yes    0
No     4
Name: (Female, Single), dtype: int32

df.loc['Female', ('Yes', 'No')] # 'Yes'selects yes values from first column level, 'No' selects from second col-level

Marital Status
Single     4
Married    3
Name: (Yes, No), dtype: int32
```
A MultiIndex can be created from a list of arrays (using MultiIndex.from_arrays()), an array of tuples (using MultiIndex.from_tuples()), a crossed set of iterables (using MultiIndex.from_product()), or a DataFrame (using MultiIndex.from_frame()).

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
[Pandas Time/Date Components](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#time-date-components)
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
## Create dummy dataframes with random data
```python
df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

# | with time index
dates = pd.date_range(start='2018-04-24', freq='D', periods=100)
df_time = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'), index=dates)
```

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
- processing
```python
# | convert column to UPPERCASE
df[col_name].str.upper()

# | count string occurence in each row
df[col_name].str.count(r'\d') # counts number of digits

# | count #chars in each row
df[col_name].str.count() # counts number of digits

# | count #tokens in each row
df[col_name].str.split().str.count() # counts number of digits

# | count #tokens in each row
df[col_name].str.split().str.count() # counts number of digits

# | split rows
s = pd.Series(["this is a regular sentence", "https://docs.python.org/3/tutorial/index.html", np.nan])
s.str.split() # splits rows by spaces (also a pattern can be used as argument). rows are now python lists with the splitted elements

0                   [this, is, a, regular, sentence]
1    [https://docs.python.org/3/tutorial/index.html]
2                                                NaN
dtype: object

s.str.split(expand=True)  # this creates new columns with the different split values (instead of lists)

s.str.rsplit("/", n=1, expand=True) # limit the number of splits to 1, and start spliting from the rights side
```

- filtering
```python
# | check if a certain word/pattern occurs in each row
df[col_name].str.contains('daada')  # returns True/False for each row

# | find occurences
df[col_name].str.findall(r'[ABC]\d') # returns a list of the found occurences of the specified pattern for each row

# | replace Weekdays by abbrevations (e.g. Monday --> Mon)
df[col_name].str.replace(r'(\w+day\b)', lambda x: x.groups[0][:3]) # () in r'' creates a group with one element, which we acces with x.groups[0]

# | create dataframe from regex groups (str.extract() uses first match of the pattern only)
df[col_name].str.extract(r'(\d?\d):(\d\d)')
df[col_name].str.extract(r'(?P<hours>\d?\d):(?P<minutes>\d\d)')
df[col_name].str.extract(r'(?P<time>(?P<hours>\d?\d):(?P<minutes>\d\d))')

# | if you want to take into account ALL matches in a row (not only first one):
df[col_name].str.extractall(r'(\d?\d):(\d\d)') # this generates a multiindex with level 1 = 'match', indicating the order of the match
```

- Replace/remove strings in df columns
```python

df[col].replace('\n', '', regex=True, inplace=True)

# remove all the characters after &# (including &#) for column - col_1
df[col].replace(' &#.*', '', regex=True, inplace=True)

# remove white space at the beginning of string 
df[col] = df[col].str.lstrip()
```

- Split string column in multiple columns using extract(regex)
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

# or: (note in the above version, its actually ONE big regex which is matched, not three different ones)
regex = (r'(?P<city>[A-Za-z ]+), (?P<state>[A-Z]{2}) (?P<zip>\d{5}(?:-\d{4})?)')  # 5-digits + optional 4-digit extension

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

