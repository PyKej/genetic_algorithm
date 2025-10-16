import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv('home-data/train.csv')

print(home_data.head())
print("----------------------------*********************----------------------------")
print(home_data.describe())