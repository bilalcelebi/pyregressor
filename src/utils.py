from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

encoder = LabelEncoder()

def get_encoding(train_df):

    train_df.dropna(axis = 1, inplace = True)
    train_cat_columns = [column for column in train_df.columns if train_df[column].dtype == 'object']

    for col in train_cat_columns:
        train_df[col] = encoder.fit_transform(train_df[col])

    
    return train_df



def get_prepared_data(train_data, target_col):

    train_df = get_encoding(train_data)
    X = train_df.drop(target_col, axis = 1)
    y = train_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=125, test_size=0.3, shuffle=True)

    return X_train, X_test, y_train, y_test
