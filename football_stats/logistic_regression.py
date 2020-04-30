import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/data.csv' , encoding='latin-1')
df = df.drop(columns = 'Unnamed: 0')

def predict(model):

    columns = ['day', 'name', 'position', 'position_level', 'age', 'age_level',
               'departure_club', 'departure_league', 'departure_country',
               'departure_continent', 'arrival_club', 'arrival_league',
               'arrival_country', 'arrival_continent', 'market_value_raw',
               'market_value', 'market_value_update', 'move_type_raw', 'move_type',
               'move_value', 'move_value_updated', 'move_year', 'move_month',
               'move_day']

    features = ['day', 'name', 'position', 'position_level', 'age', 'age_level',
                'departure_club', 'departure_league', 'departure_country',
                'departure_continent', 'arrival_club', 'arrival_league',
                'arrival_country', 'arrival_continent', 'move_type_raw', 'move_type',
                'move_value', 'move_value_updated', 'move_year', 'move_month',
                'move_day']

    label = ['market_value']

    data = df[columns].dropna().reset_index(drop=True)

    X = data[features].dropna().reset_index(drop=True)
    Y = data[label].dropna().reset_index(drop=True)

    categorical_feature_mask = X.dtypes == object
    categorical_cols = X.columns[categorical_feature_mask].tolist()

    le = LabelEncoder()
    X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col))

    x_train, x_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(Y, test_size=0.2, random_state=42)

    regr = linear_model.LinearRegression()

    # fit the model
    regr.fit(np.array(x_train), np.array(y_train))

    y_pred = regr.predict(np.array(x_test))

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(np.array(y_test.values), y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(np.array(y_test), y_pred))

    return regr.coef_

def parse_args():

    parser = argparse.ArgumentParser(
        description="Transfer value prediction"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="logistic-regression",
        help="""
        Project
        """,
    )

    return parser.parse_args()


if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    argv = parse_args()

    predict(argv)