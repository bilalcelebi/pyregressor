from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import numpy as np
import pandas as pd
import os
import pickle

models = {
    "Linear Regression":LinearRegression(),
    "Ridge":Ridge(),
    "ElasticNet":ElasticNet(),
    "Lasso":Lasso(),
    "Decision Tree Regressor":DecisionTreeRegressor(),
    "Extra Tree Regressor":ExtraTreeRegressor()
}


def get_model_result(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(preds, y_test)
    mse = np.sqrt(mse)
    r2 = r2_score(preds, y_test)
    mae = mean_absolute_error(preds, y_test)

    scores = {
        'MSE':mse,
        'MAE':mae,
        'R2':r2
    }

    return scores



def get_results(X_train, y_train, X_test, y_test, to_save):

    all_results = dict()

    for model in models.keys():

        results = get_model_result(models[str(model)], X_train, y_train, X_test, y_test)
        all_results[str(model)] = results

    sorted_results = sorted(all_results.items(), key = lambda x: x[1]['MSE'], reverse = False)
    best_model = sorted_results[0][0]

    fitted_model = models[best_model]
    fitted_model.fit(X_train, y_train)

    save_path = str(os.getcwd())
    file_path = os.path.join(save_path, f'{best_model}_fitted.sav')


    if bool(to_save) == True:
        pickle.dump(fitted_model, open(file_path, "wb"))


    return sorted_results, file_path 