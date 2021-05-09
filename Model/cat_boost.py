import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool

def train_catboost(x_train, x_test, y_train, y_test):
    cat_boost = CatBoostClassifier(
        iterations=500,
        random_seed=42,
        loss_function='MultiClass',
        task_type='GPU',
        learning_rate=0.2,
        custom_metric= ['Accuracy'],
    )
    cat_boost.fit(
        x_train,
        y_train,
        eval_set=(x_test, y_test),
    )
    return cat_boost


X = np.load('features.npy')
Y = np.load('age.npy')

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)

save_name = 'ad'
cat_boost = train_catboost(x_train, x_test, y_train, y_test)
eval_result = cat_boost.eval_metrics(Pool(x_test,y_test), ['Accuracy'])
joblib.dump(eval_result,save_name)
# preds = cat_boost.predict(x_test)

