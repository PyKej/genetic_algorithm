import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv('home-data/train.csv')

# print(home_data.head())
# print("----------------------------*********************----------------------------")
# print(home_data.describe())


y = home_data.SalePrice

features = [
    'MSSubClass',
    'LotArea',
    'OverallQual',
    'OverallCond',
    'YearBuilt',
    'YearRemodAdd',
    '1stFlrSF',
    '2ndFlrSF',
    'LowQualFinSF',
    'GrLivArea',
    'FullBath',
    'HalfBath',
    'BedroomAbvGr',
    'KitchenAbvGr',
    'TotRmsAbvGrd',
    'Fireplaces',
    'WoodDeckSF',
    'OpenPorchSF',
    'EnclosedPorch',
    '3SsnPorch',
    'ScreenPorch',
    'PoolArea',
    'MiscVal',
    'MoSold',
    'YrSold'
]

X = home_data[features]

# test size will be 20%. random state - start alwais from same seed (not random in iterations)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=69) 

rf_model = RandomForestRegressor(random_state=69)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print(rf_val_predictions)
print("------------------")
print(val_y)
print("Validation MAE for Random Forest Model: {:,.2f}".format(rf_val_mae))
