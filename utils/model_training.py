import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV

def train_xgboost_model(X_train, y_train):
    """
    Trains an XGBoost model on the training set using random search with cross-validation.

    Parameters:
    - X_train (pd.DataFrame): The training features.
    - y_train (pd.Series): The training labels.

    Returns:
    - model (xgb.XGBClassifier): The trained XGBoost model with the best hyperparameters.
    """

    # Define the parameter grid for random search
    param_grid = {
        'max_depth': [3, 6]
    }

    # Create the XGBoost classifier
    model = xgb.XGBClassifier(eval_metric='error', use_label_encoder=False)

    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=3, cv=3, scoring='f1_macro', random_state=42)

    # Fit the RandomizedSearchCV object on the training data
    random_search.fit(X_train, y_train)

    # Retrieve the best model
    model = random_search.best_estimator_

    return model


def calculate_f1_score(model, X_test, y_test):
    """
    Calculates the F1 score on the testing set using the trained XGBoost model.

    Parameters:
    - model (xgb.XGBClassifier): The trained XGBoost model.
    - X_test (pd.DataFrame): The testing features.
    - y_test (pd.Series): The testing labels.

    Returns:
    - f1 (float): The F1 score on the testing set.
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    return f1