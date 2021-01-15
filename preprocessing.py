import numpy as np, pandas as pd, pandas_profiling as profile
from itertools import product
from sklearn.preprocessing import LabelEncoder

items = pd.read_csv('./dataset/items.csv')
shops = pd.read_csv('./dataset/shops.csv')
categories = pd.read_csv('./dataset/item_categories.csv')
train = pd.read_csv('./dataset/sales_train.csv')
# set index to ID to avoid dropping it later
test = pd.read_csv('./dataset/test.csv').set_index('ID')

# 95% percentile of price is 2690, 95% percentile of sales is 2
report = profile.ProfileReport(train)
report.to_file('EDA')

# remove outliers
train = train[(train.item_price >= 0) & (train.item_price <= 2690)]
train = train[train.item_cnt_day <= 10]
mean = train[(train.shop_id == 32) & (train.item_id == 2973) & (train.date_block_num == 4) & (train.item_price > 0)].item_price.mean()
train.loc[train.item_price < 0, 'item_price'] = mean  # there is one item with price below zero

# duplicates shop id, 0 = 57, 1 = 58, 10 = 11
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

# each shop_name starts with the city name, each category contains type and subtype in its name
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id','city_code']]

categories['split'] = categories['item_category_name'].str.split('-')
categories['type'] = categories['split'].map(lambda x: x[0].strip())
categories['type_code'] = LabelEncoder().fit_transform(categories['type'])
# if subtype is nan then type
categories['subtype'] = categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
categories['subtype_code'] = LabelEncoder().fit_transform(categories['subtype'])
categories = categories[['item_category_id','type_code', 'subtype_code']]

items.drop(['item_name'], axis=1, inplace=True)

matrix = []
# cartesian product by 'date_block_num', 'shop_id', 'item_id'
cols = ['date_block_num', 'shop_id', 'item_id']
for i in range(34):
	sales = train[train.date_block_num == i]
	matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols, inplace=True)

train['revenue'] = train['item_price'] * train['item_cnt_day']
# count monthly sales
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = matrix['item_cnt_month'].fillna(0).clip(0,20).astype(np.float16)

# concatenate train and test
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 months

# merge city and category features
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, categories, on=['item_category_id'], how='left')
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)

# past [lags] months records
def lag_feature(df, lags, col):
	tmp = df[['date_block_num','shop_id','item_id',col]]
	for i in lags:
		shifted = tmp.copy()
		shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
		shifted['date_block_num'] += i
		df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
	return df

# average sales in one month
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)

# average sales for every product before one, two, three, six and twelve months
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_item_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

# average sales for every shop before one, two, three, six and twelve months
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

# average sales for every category before one month
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_cat_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)

# average sales for the same shop and category before one month
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_cat_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)

# average sales for the same shop and type before one month
group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_type_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)

# average sales for the same shop and subtype before one month
group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_subtype_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)

# average sales for every city before one month
group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_city_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

# average sales for the same product and city before one month
group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_item_city_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

# average sales for every type before one month
group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_type_avg_item_cnt']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)

# average sales for every subtype before one month
group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_subtype_avg_item_cnt' ]
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')
matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)

# average price
group = train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

# average price before one, two, three, four, five and six months
group = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)
lags = [1,2,3,4,5,6]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

# price trend in six months
for i in lags:
	matrix['delta_price_lag_'+str(i)] = (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']
# only keep one valid delta_price_lag
def select_price_trend(row):
	for i in lags:
		if row['delta_price_lag_' + str(i)]:
			return row['delta_price_lag_' + str(i)]
	return 0
matrix['delta_price_lag'] = matrix.apply(select_price_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)
fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
	fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
	fetures_to_drop += ['delta_price_lag_'+str(i)]
matrix.drop(fetures_to_drop, axis=1, inplace=True)

# monthly average revenue of every shop
group = train.groupby(['date_block_num','shop_id']).agg({'revenue': ['mean']})
group.columns = ['date_shop_avg_revenue']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_revenue'] = matrix['date_shop_avg_revenue'].astype(np.float32)
matrix = lag_feature(matrix, lags, 'date_shop_avg_revenue')
# average revenue of every shop
group = train.groupby(['shop_id']).agg({'revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)
# revenue trend in six months
for i in lags:
	matrix['delta_revenue_lag_'+str(i)] = (matrix['date_shop_avg_revenue_lag_'+str(i)] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
# only keep one valid delta_revenue_lag
def select_revenue_trend(row):
	for i in lags:
		if row['delta_revenue_lag_' + str(i)]:
			return row['delta_revenue_lag_' + str(i)]
	return 0
matrix['delta_revenue_lag'] = matrix.apply(select_revenue_trend, axis=1)
matrix['delta_revenue_lag'] = matrix['delta_revenue_lag'].astype(np.float16)
matrix['delta_revenue_lag'].fillna(0, inplace=True)
fetures_to_drop = ['date_shop_avg_revenue', 'shop_avg_revenue']
for i in lags:
	fetures_to_drop += ['date_shop_avg_revenue_lag_'+str(i)]
	fetures_to_drop += ['delta_revenue_lag_'+str(i)]
matrix.drop(fetures_to_drop, axis=1, inplace=True)

# month gap between the last transaction and current transaction for the same shop and product
cache = {}
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():
	key = str(row.item_id)+' '+str(row.shop_id)
	if key not in cache:
		if row.item_cnt_month!=0:
			cache[key] = row.date_block_num
	else:
		last_date_block_num = cache[key]
		matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
		cache[key] = row.date_block_num

# month gap between the last transaction and current transaction for the same product
cache = {}
matrix['item_last_sale'] = -1
matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():
	key = row.item_id
	if key not in cache:
		if row.item_cnt_month!=0:
			cache[key] = row.date_block_num
	else:
		last_date_block_num = cache[key]
		if row.date_block_num>last_date_block_num:
			matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
			cache[key] = row.date_block_num

# month gap between the first transaction and current transaction for the same shop and product
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
# month gap between the first transaction and current transaction for the same product
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')

# month twelve lag features were used, so no need for those first eleven month records
matrix = matrix[matrix.date_block_num > 11]

# lots of lag features are full of nan, so fill them with zero
def fill_na(df):
	for col in df.columns:
		if ('_lag_' in col) & (df[col].isnull().any()):
			if 'item_cnt' in col:
				df[col].fillna(0, inplace=True)
	return df
matrix = fill_na(matrix)

matrix.to_csv('./dataset/clean_data',index=False)