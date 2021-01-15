import pandas as pd, matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):
	fig, ax = plt.subplots(1,1,figsize=figsize)
	return plot_importance(booster=booster, ax=ax)

data = pd.read_csv('./dataset/clean_data')
# 34 month for the test set, 33 month for the validation set and 12-33 months for the train set
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
del data

model = XGBRegressor(max_depth=8, n_estimators=1000, min_child_weight=300, colsample_bytree=0.8, subsample=0.8, eta=0.3)
model.fit(X_train, Y_train, eval_metric="rmse", eval_set=[(X_train, Y_train), (X_valid, Y_valid)], verbose=True, early_stopping_rounds = 10)
plot_features(model, (14,14))