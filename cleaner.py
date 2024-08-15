import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from catboost import CatBoostClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def load_and_prepare_data(filepath):
    try:
        df = pd.read_csv(filepath, sep='\t', header=6, skiprows=[7], engine='python')
    except (pd.errors.ParserError, FileNotFoundError) as e:
        print(f"Error loading data: {e}")
        return None

    # Remove 'label' and 'det_label' columns
    df.drop(['label', 'det_label'], axis=1, inplace=True, errors='ignore')

    return df

def preprocess_features(df):
    # Determine numeric and categorical features based on data types
    numeric_features = df.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Impute missing values in numeric columns using KNNImputer
    numeric_imputer = KNNImputer(n_neighbors=5)
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

    # Impute missing values in categorical columns using KNeighborsClassifier
    categorical_imputer = KNeighborsClassifier(n_neighbors=5, weights='distance')
    for cat in categorical_features:
        encoder = LabelEncoder()
        encoded_values = encoder.fit_transform(df[cat])
        imputed_values = categorical_imputer.fit(df[numeric_features], encoded_values).predict(df[numeric_features])

        # Replace None values with a default value or a new category label
        default_value = 'missing'  # or any other suitable value
        imputed_values = [value if value is not None else default_value for value in encoder.inverse_transform(imputed_values)]

        df[cat] = imputed_values

    # Define categorical transformer
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Handle categorical features for CatBoost
    X_categorical = categorical_transformer.fit_transform(df[categorical_features])
    X_numeric = df[numeric_features]
    X = np.column_stack((X_numeric, X_categorical))

    # Ensure X is a 2D array
    X = X.reshape(-1, X.shape[-1])

    # Add a target variable (y) for classification
    target_column_name = 'tunnel_parents'  # Replace with the actual name of your target column
    y = df[target_column_name]

    return X, y

def train_initial_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # Ensure y_train and y_test are 1D arrays
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Train the initial model with default hyperparameters
    initial_model = CatBoostClassifier(verbose=200)
    initial_model.fit(X_train, y_train)

    # Evaluate the initial model on the test set
    y_pred = initial_model.predict(X_test)
    initial_accuracy = np.mean(y_pred == y_test)
    print(f"Initial model accuracy on test set: {initial_accuracy * 100:.2f}%")

    return initial_model, X_train, y_train, X_test, y_test, initial_accuracy

def optimize_hyperparameters(initial_model, X_train, y_train, max_evals=20):
    # Define the objective function
    def objective_function(params):
        iterations = int(params['iterations'])
        learning_rate = params['learning_rate']
        depth = int(params['depth'])

        model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0)

        # Perform K-fold cross-validation on the training set
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        mean_score = cv_scores.mean()

        # Hyperopt minimizes the objective function, so we return the negative mean score
        return {'loss': -mean_score, 'status': STATUS_OK}

    # Define the search space
    search_space = {
        'iterations': hp.quniform('iterations', 500, 2000, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.1), np.log(0.9)),  # Higher range for learning rate
        'depth': hp.quniform('depth', 4, 10, 1)
    }

    # Optimize hyperparameters using Hyperopt
    trials = Trials()
    best_params = fmin(objective_function, search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    # Print the best hyperparameters
    print("Best hyperparameters:")
    print(best_params)

    return best_params

def train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params):
    iterations = int(best_params['iterations'])
    learning_rate = best_params['learning_rate']
    depth = int(best_params['depth'])

    model = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=200)
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = model.predict(X_test)
    optimized_accuracy = np.mean(y_pred == y_test)
    print(f"Optimized model accuracy on test set: {optimized_accuracy * 100:.2f}%")

    return optimized_accuracy

if __name__ == "__main__":
    data_file = 'conn.log.labeled.txt'
    df = load_and_prepare_data(data_file)
    if df is not None:
        X, y = preprocess_features(df)

        # Train the initial model with default hyperparameters
        initial_model, X_train, y_train, X_test, y_test, initial_accuracy = train_initial_model(X, y)

        # Optimize hyperparameters using Hyperopt
        best_params = optimize_hyperparameters(initial_model, X_train, y_train)

        # Train and evaluate the model with optimized hyperparameters
        optimized_accuracy = train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params)

        # Calculate and print the percentage difference in accuracy
        accuracy_difference = (optimized_accuracy - initial_accuracy) / initial_accuracy * 100
        print(f"Percentage difference in accuracy: {accuracy_difference:.2f}%")
    else:
        print("Failed to load data.")















