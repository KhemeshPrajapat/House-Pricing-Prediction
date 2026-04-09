import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose  import ColumnTransformer
from sklearn.linear_model  import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics  import mean_squared_error
from sklearn.model_selection import cross_val_score

housing =  pd.read_csv("housing.csv")

housing["income_category"] = pd.cut(housing["median_income"],
                                     bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                     labels = [1,2,3,4,5])

data_split =  StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in data_split.split(housing, housing["income_category"]):
    strata_train_set = housing.loc[train_index].drop("income_category", axis = 1)
    strata_test_set = housing.loc[test_index].drop("income_category", axis = 1)

housing = strata_train_set.copy()

housing_labels = housing['median_house_value'].copy()
housing = housing.drop(["median_house_value"], axis = 1)


num_attributes = housing.drop('ocean_proximity', axis = 1).columns.to_list()
cat_attributes = ['ocean_proximity']

num_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
   ( "numerical", num_pipeline, num_attributes),
   ("categorical",cat_pipeline,cat_attributes)
])

housing_prepared = full_pipeline.fit_transform(housing)


linear_reg = LinearRegression()
linear_reg.fit(housing_prepared, housing_labels)
linear_predict = linear_reg.predict(housing_prepared)
linear_RMSE = mean_squared_error(housing_labels, linear_predict,    )
print(F"The RMSE of linear model is {linear_RMSE}")


decision_reg = DecisionTreeRegressor(random_state=42)
decision_reg.fit(housing_prepared, housing_labels)
decision_predict = decision_reg.predict(housing_prepared)
decision_RMSE = -cross_val_score(decision_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(f"The crosss validation of the linear model is {linear_RMSE}")
print(pd.DataFrame(decision_RMSE).describe())


random_reg =  RandomForestRegressor(random_state=42)
random_reg.fit(housing_prepared, housing_labels)
random_predict = random_reg.predict(housing_prepared)
random_RMSE = mean_squared_error(housing_labels, random_predict,    )
print(f"The RMSE of the random model is {random_RMSE}")