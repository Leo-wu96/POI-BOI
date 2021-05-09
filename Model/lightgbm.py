# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 22:28:11 2021

@author: yxk
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.model_selection import train_test_split

#将list转为dataframe
def changetodata(lists):
    newlist=[int(s[4:]) for s in lists]
    df=pd.DataFrame(newlist, columns=['user_id']) 
    return df


def SaveFeatureData(UserDocDist,modelname):
    
    # Save
    print("保存文件中")
    np.save('./data/data4/UserFeature_{}'.format(modelname), UserDocDist)
 
    
#lgb模型
def lgbmodel(X_train,y_train,X_test,Y_test,num_class,name):
    '''
    lightGbm模型的训练
    ''' 
    param = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'nthread':40,
        'metric': ['multi_logloss', 'multi_error','average_precision'],
        'silent': True,#是否打印信息，默认False
        'learning_rate': 0.01,
        'num_class':num_class,
        'num_leaves': 128,
        'max_depth': 6,
        'max_bin': 127,
        'subsample_for_bin': 50000,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 1,
        'reg_lambda': 0,
        'min_split_gain': 0.0,
        'min_child_weight': 1,
        'min_child_samples': 20,
        'scale_pos_weight': 1,
        'device':'gpu'
    }
     
    trn_data = lgb.Dataset(X_train, y_train)
    test_data = lgb.Dataset(X_test,Y_test, reference=trn_data)
    num_round = 50
    
    evals_result = {}
    clf = lgb.train(param, 
                    trn_data, 
                    num_round, 
                    valid_sets=test_data,
                    early_stopping_rounds=10,
                    evals_result=evals_result, # stores validation results.
                    verbose_eval = 10)
    joblib.dump(evals_result,'./evals_result.pkl')

    predictions= clf.predict(X_test, num_iteration=clf.best_iteration)
    print('Plotting metrics recorded during training...')
    ax = lgb.plot_metric(evals_result, metric='multi_error')
    plt.show()   
    
    """
    保存模型和参数
    """
    # 模型保存
    clf.save_model('./data_one/lgbmodel/lgbmodel_{}.txt'.format(name))

    
    return clf


if __name__ == "__main__":
    #训练集
    X_path = 'features.npy'
    Y_path = 'age.npy'
    X=np.load(X_path)
    Y=np.load(Y_path)
    
    
    """
    读取测试集
    10万测试集       
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10,random_state=42)
    
    
    """
    训练lgb模型
    """
    num_class=10
    name="feature_creative"
    res=lgbmodel(X_train,Y_train,X_test,Y_test,num_class,name)