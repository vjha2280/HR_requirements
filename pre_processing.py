import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import numpy as np

def read_clean():
    train = pd.read_csv('aug_train.csv')
    test = pd.read_csv('aug_test.csv')
    df = pd.concat([train, test])
    # Data cleaning
    df['company_size'] = df['company_size'].replace('Oct-49', '10-49')
    df['company_size'] = df['company_size'].replace('10/49', '10-49')
    df['company_size'] = df['company_size'].replace('<10', '1-9')
    df['company_size'] = df['company_size'].replace('10000+', '10000-15000')
    df['last_new_job'] = df['last_new_job'].replace('never', 0)
    df['last_new_job'] = df['last_new_job'].replace('>4', 0.5)
    df['experience'] = df['experience'].replace('<1', 0)
    df['experience'] = df['experience'].replace('>20', 21)
    return df

#define mapping for columns unique values
def find_category_mappings(df,variable):
    return {k: i for i, k in enumerate(df[variable].dropna().unique(), 0)}

# Ordinal encoding for mapped columns
def integer_encode(df , variable, ordinal_mapping):
    df[variable] = df[variable].map(ordinal_mapping)

mm = MinMaxScaler()
mappin = dict()
def imputation(df1, cols):
    df = df1.copy()
    for variable in cols:
        mappings = find_category_mappings(df, variable)
        mappin[variable] = mappings
    for variable in cols:
        integer_encode(df, variable, mappin[variable])
    sca = mm.fit_transform(df)
    knn_imputer = KNNImputer()
    knn = knn_imputer.fit_transform(sca)
    df.iloc[:, :] = mm.inverse_transform(knn)
    for i in df.columns:
        df[i] = round(df[i]).astype('int')
    #Inverse transform
    for i in cols:
        inv_map = {v: k for k, v in mappin[i].items()}
        df[i] = df[i].map(inv_map)
    return df

def missing_value_handling(df):
    #creat dummy for not null columns
    df_dummy = pd.get_dummies(data=df[['city', 'relevent_experience']])
    df = df.drop(['city', 'relevent_experience'], axis=1)
    df = pd.concat([df, df_dummy], axis=1)
    df['experience'] = pd.to_numeric(df['experience'])
    df['last_new_job'] = pd.to_numeric(df['last_new_job'])
    # missing value imputation for numeric columns
    av_column = df.mean(axis=0)
    df['experience'] = df['experience'].fillna(av_column['experience'])
    df['last_new_job'] = df['last_new_job'].fillna(av_column['last_new_job'])
    # missing values imputation for categorical columns which has null values
    col_list = ['gender', 'company_size', 'company_type', 'education_level', 'enrolled_university','major_discipline']
    df_new = df[col_list] #copy columns which have null values in new df and drop them
    df = df.drop(col_list, axis=1)
    df = df.drop(['target'], axis=1)
    for variable in df_new.columns:
        df = pd.concat([df, df_new[[variable]]], axis=1)
        df1 = imputation(df, [variable])
        print('Imputation done for :',variable)
        df2 = pd.get_dummies(data=df1[[variable]])
        df = df.drop([variable], axis=1)
        df = pd.concat([df, df2], axis=1)
    return df

if __name__ == "__main__":
    df = read_clean()
    print('Errors has been removed from the raw data')
    df_orig = df.copy(deep=True)
    print('Clean data backup')
    df = missing_value_handling(df)
    print('Missings values has been impute')
    df = pd.concat([df, df_orig[['target']]], axis=1)
    df.to_csv('clean_data.csv',index=False)