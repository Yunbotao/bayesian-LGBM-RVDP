from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error   
from scipy.stats import pearsonr
import lightgbm as lgb
import warnings
import shap as shap
import matplotlib.pyplot as plt      
import joblib
import matplotlib.ticker as ticker
import csv

warnings.filterwarnings("ignore", category=UserWarning)
lgb.register_logger(lgb.basic._DummyLogger())

for a in range(7,13,1):
    for i in range(1,11,1):

        data = pd.read_csv(r"xxx\data.csv",index_col=0)  
        scaler = MinMaxScaler() 
        scaler = scaler.fit(data)      
        data1=scaler.transform(data)
        pd.DataFrame(data1)
        X = data1[:, 0:7]
        y = data1[:, a]


        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

        fitness_history = []
        pingjiazhibiao = []

        

        def black_box_function(
        num_leaves, 
        max_depth, 
        learning_rate, 
        n_estimators, 
        min_data_in_leaf, 
        lambda_l1, 
        lambda_l2,
        feature_fraction, 
        bagging_fraction,
        bagging_freq,
        max_cat_group,
        min_split_gain
        ):

            lgb = LGBMRegressor(
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            n_estimators=int(n_estimators),
            min_data_in_leaf=int(min_data_in_leaf),
            lambda_l1=float(lambda_l1),
            lambda_l2=float(lambda_l2),
            feature_fraction=float(feature_fraction),
            bagging_fraction=float(bagging_fraction),
            bagging_freq=int(bagging_freq),
            max_cat_group=int(max_cat_group),
            min_split_gain=float(min_split_gain),
            metric='l2',
            boosting_type='gbdt',
            verbose = -1
            )

            score = cross_val_score(lgb, X_train, Y_train, cv=5).mean()
            fitness_history.append(score)
            return score

        pbounds = {
        'num_leaves': (20, 100),
        'max_depth': (-1, 100),
        'learning_rate': (0.001, 0.1),
        'n_estimators': (10, 1000),
        'min_data_in_leaf': (5, 100),
        'lambda_l1': (0, 0.015),
        'lambda_l2': (0, 0.015),
        'feature_fraction': (0.1, 1),
        'bagging_fraction': (0.1, 1),
        'bagging_freq': (0, 10),
        'max_cat_group': (1, 200),
        'min_split_gain': (0, 0.1),       
        }

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2,
            random_state=1
        )

        optimizer.maximize(
            init_points= 100,  
            n_iter= 500   
            
        )

        print("最佳参数:", optimizer.max['params'])
        print("最佳目标（得分）:", optimizer.max['target'])
        best_params = optimizer.max['params']
        best_model = LGBMRegressor(
        num_leaves=int(best_params['num_leaves']),
        max_depth=int(best_params['max_depth']),
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        min_data_in_leaf=int(best_params['min_data_in_leaf']),
        lambda_l1=best_params['lambda_l1'],
        lambda_l2=best_params['lambda_l2'],
        feature_fraction=best_params['feature_fraction'],
        bagging_fraction=best_params['bagging_fraction'],
        bagging_freq=int(best_params['bagging_freq']),
        max_cat_group=int(best_params['max_cat_group']),
        min_split_gain=best_params['min_split_gain'],
        metric='l2',
        boosting_type='gbdt',
        verbose = -1
        )

        best_model.fit(X_train, Y_train)
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        file_name_1 = f"贝叶斯-LGB0_{a}_{i}.joblib"
        joblib.dump(best_model,file_name_1)
        
        
        print("r2_train:", r2_score(Y_train, y_train_pred))
        print("MSE_train:", mean_squared_error(Y_train, y_train_pred))
        print("RMSE_train:", np.sqrt(mean_squared_error(Y_train, y_train_pred)))
        print("MAE_train:", mean_absolute_error(Y_train, y_train_pred))
        print("R_train:", pearsonr(Y_train, y_train_pred)[0])

        print("r2_test:", r2_score(Y_test, y_test_pred))
        print("MSE_test:", mean_squared_error(Y_test, y_test_pred))
        print("RMSE_test:", np.sqrt(mean_squared_error(Y_test, y_test_pred)))
        print("MAE_test:", mean_absolute_error(Y_test, y_test_pred))
        print("R_test:", pearsonr(Y_test, y_test_pred)[0])
        
        base_path = 'xxx'
 
        explainer = shap.Explainer(best_model.predict,X_train) 
        shap_values = explainer(X_train) 
        
        feature_label = ['f','b','g','c','h','d','i']

        plt.rcParams.update({'font.family': 'Times New Roman'})
        plt.rcParams['font.size'] = 20
    
        filename_5 = f'{base_path}beiyesi-LGB-汇总图_{a}_{i}.png'
        shap.summary_plot(shap_values,feature_names=feature_label, show = False)
        plt.savefig(filename_5,dpi=1200, bbox_inches='tight')
        plt.show()
        
        filename_2 = f'{base_path}beiyesi-LGB-summary_plot_{a}_{i}.png'
        shap.summary_plot(shap_values,feature_names=feature_label,plot_type = "bar",show = False)
        plt.savefig(filename_2,dpi=1200, bbox_inches='tight')
        plt.show()
        
        mean_abs_shap = shap_values.abs.mean(0)
        importance_series = pd.Series(mean_abs_shap.values, index=feature_label).sort_values(ascending=False)
        importance_series.to_csv(f"{base_path}beiyesi-LGB-summary_plot_{a}_{i}.csv", header=["Mean_ABS_SHAP"])
        
        print("r2_train:", r2_score(Y_train, y_train_pred))
        print("MSE_train:", mean_squared_error(Y_train, y_train_pred))
        print("RMSE_train:", np.sqrt(mean_squared_error(Y_train, y_train_pred)))
        print("MAE_train:", mean_absolute_error(Y_train, y_train_pred))
        print("R_train:", pearsonr(Y_train, y_train_pred)[0])

        print("r2_test:", r2_score(Y_test, y_test_pred))
        print("MSE_test:", mean_squared_error(Y_test, y_test_pred))
        print("RMSE_test:", np.sqrt(mean_squared_error(Y_test, y_test_pred)))
        print("MAE_test:", mean_absolute_error(Y_test, y_test_pred))
        print("R_test:", pearsonr(Y_test, y_test_pred)[0])
        
        pingjiazhibiao.append(r2_score(Y_train, y_train_pred))
        pingjiazhibiao.append(mean_squared_error(Y_train, y_train_pred))
        pingjiazhibiao.append(np.sqrt(mean_squared_error(Y_train, y_train_pred)))
        pingjiazhibiao.append(mean_absolute_error(Y_train, y_train_pred))
        pingjiazhibiao.append(pearsonr(Y_train, y_train_pred)[0])
        

        pingjiazhibiao.append(r2_score(Y_test, y_test_pred))
        pingjiazhibiao.append(mean_squared_error(Y_test, y_test_pred))
        pingjiazhibiao.append(np.sqrt(mean_squared_error(Y_test, y_test_pred)))
        pingjiazhibiao.append(mean_absolute_error(Y_test, y_test_pred))
        pingjiazhibiao.append(pearsonr(Y_test, y_test_pred)[0])
               
        filename_3 = f'{base_path}评价指标_{a}_{i}.csv'
        np.savetxt(filename_3, pingjiazhibiao, delimiter=',')
             
        filename_4 = f'{base_path}X_train_{a}_{i}.csv'  
        np.savetxt(filename_4, X_train, delimiter=',')
        
        filename_5 = f'{base_path}X_test_{a}_{i}.csv'  
        np.savetxt(filename_5, X_test, delimiter=',')
        
        filename_6 = f'{base_path}Y_train_{a}_{i}.csv'  
        np.savetxt(filename_6, Y_train, delimiter=',')
        
        filename_7 = f'{base_path}Y_test_{a}_{i}.csv'  
        np.savetxt(filename_7, Y_test, delimiter=',')
        
        filename_8 = f'{base_path}beiyesi-randomforest-_fitness_history_{a}_{i}.csv' 
        np.savetxt(filename_8, fitness_history, delimiter=',')
        
        