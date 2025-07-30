import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data =pd.read_csv('train.csv')

Y = data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = data[features]

train_X , train_Y , val_X , val_Y = train_test_split(X,Y, random_state = 1)
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X,train_Y)


val_prediction = model.predict(val_X)

val_mae = mean_absolute_error(val_y, val_predictions)
print("Validation MAE:", val_mae)