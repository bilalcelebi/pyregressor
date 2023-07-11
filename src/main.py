import os
import pandas as pd
from utils import get_prepared_data
from regressors import get_results
import argparse



def get_data(data_path):

    df = pd.read_csv(data_path)
    df.dropna(axis = 1, inplace = True)
    forbidden_cols = ['id','ID']

    for column in df.columns:
        if column in forbidden_cols:
            df.drop(column, axis = 1, inplace = True)

    return df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required = True)
    parser.add_argument("--target_column", required = True)
    parser.add_argument("--save_model", required = True)

    args = parser.parse_args()

    data = get_data(data_path = args.data_path)
    target_column = str(args.target_column)
    model_path = str(args.save_model)


    X_train, X_test, y_train, y_test = get_prepared_data(data, target_column)
    results, save_path = get_results(X_train, y_train, X_test, y_test, bool(model_path))

    print(results)
    print('\n')
    print(f'Saved Model Path : {save_path}')


if __name__ == '__main__':
    main()