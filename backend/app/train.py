# training the model using the xgboost regressor 

import pandas as pd 
from xgboost import XGBRegressor 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pre_sales = pd.read_csv("artifacts/pre_sales.csv")

# target is "sales"
# features are "yesterday_sales", "last_week_sales", "week_sales_mean"

features = ["yesterday_sales", "last_week_sales", "week_sales_mean"]
target = "Sales"

X = pre_sales[features]
y = pre_sales[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle = False, test_size =0.2)

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05
)

# fitting the model, predicting the target value for the test set and 
# calculating the mean absolute error 

model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print("MAE:", mae)
