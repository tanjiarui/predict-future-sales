# Practice of Feature Engineering

## Dataset

data description is [here](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)

## Work Pipeline

1. Run preprocessing.py to complete the feature construction, combination and merge. Each code snippet has been commented.
2. Run model.py to evaluate feature importance.

## Some Introspection

1. Features are priority.
2. It is always necessary to generate some new features based on data at hand. The principle to construct these new features is from either business insights or industrial characteristics. For instance, retail industry is all about people, shop and product. The logic to make new features depends on what kind of object that the model is about to evaluate. If the model needs to predict sales, every record should be product-centric in the dataframe. If the model needs to predict re-purchase propensity, every record should be customer-centric in the dataframe.
3. Add operation is for feature preparation, while subtraction operation is for modeling. In other words, we should discover as much more valuable features as possible at the first step. Then, analyzing feature importance and taking feature selection or feature reduction to remove less useful features.
