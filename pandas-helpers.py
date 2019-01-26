import pandas as pd
import numpy as np


""" FILTERING """

def df_drop_multiple_columns(col_names_list, df): 
    df.drop(col_names_list, axis=1, inplace=True)
    return df


""" TRANSFORMATION """
def df_convert_timestamp(df): 
    """ Convert timestamp(from string to datetime format) """
    df.insert(loc=2, column='timestamp', value=pd.to_datetime(df.transdate, format='%Y-%m-%d %H:%M:%S.%f')) 


""" CONCATENATION """

def df_concat_col_str_condition(df):
    """ Concat two columns with strings if the last 3 letters of the first column are 'pil' 
     For instance, you want to concatenate the 1st column with the 2nd column if the strings 
     in the 1st column end with certain letters."""
    mask = df['col_1'].str.endswith('pil', na=False)
    col_new = df[mask]['col_1'] + df[mask]['col_2']
    col_new.replace('pil', ' ', regex=True, inplace=True)  # replace the 'pil' with emtpy space


""" STATS """

def df_check_missing_data(df):
    """ check the number of missing data for each column """
    # check for any missing data in the df (display in descending order)
    return df.isnull().sum().sort_values(ascending=False)


""" FUNCTIONS FOR CATEGORICAL COLUMNS """

def df_split_cat_and_num_colums(df):
    df_cat = df.select_dtypes(include=['object']).copy()
    df_num = df.drop(df.columns, axis=1, inplace=True) # del df[['c', 'd']]

    return df_cat, df_num


def df_get_categorical_feature_names(df):
    df_cat = df.select_dtypes(include=['object'])
    return df_cat.columns.tolist()


def create_categorical_feature_mapping(df, keep_nan=True):
    """ Returns a dictionary that maps nominal values of categorical features to integer values """
    cat_colname_list = df_get_categorical_feature_names(df)
    cat_mapping_dict = dict()
    for cat_name in cat_colname_list:
        nominal_to_code_dict = dict()
        df[cat_name] = df[cat_name].astype('category') # convert dtypte to use df[cat_name].cat.codes
        df_cat_and_code = pd.DataFrame({'nominal': df[cat_name].values, 'numerical': df[cat_name].cat.codes.values}).drop_duplicates()
        df_cat_and_code['numerical'] = df_cat_and_code['numerical'].astype('int32')
        for _, row in df_cat_and_code.iterrows():
            if keep_nan and code == -1:
                code = np.nan
            nominal_to_code_dict[row['nominal']] = row['numerical']
        
        cat_mapping_dict[cat_name] = nominal_to_code_dict

    return cat_mapping_dict


def df_categorical_to_numerical_with_dict(df):
    """ Map the categorical nominal values to integer values using a dictionary that defines the mapping.
    It is important to define such a dictionary when always the same mapping is to me used.
    (Would be best to save the cat_mapping_dict in a file, and load it from there)"""
    cat_mapping_dict = create_categorical_feature_mapping(df)

    return df.replace(cat_mapping_dict, inplace=True)


def df_categorical_to_numerical(df):
    """ Map the categorical nominal values to integer values"""

    cat_col_names_list = df_get_categorical_feature_names(df)
    df[cat_col_names_list] = df[cat_col_names_list].astype('category')

    for col_name in cat_col_names_list:
        df[col_name] = df[col_name].cat.codes

    return df


