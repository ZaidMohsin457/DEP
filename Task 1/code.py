import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the data
data_path = 'data/Ames_Housing_Data.tsv'
data = pd.read_csv(data_path, delimiter='\t')

# Step 1: Data Cleaning and Preprocessing

# Define features and target
target = 'SalePrice'
features = data.columns.drop(['Order', 'PID', target])

# Separate numerical and categorical features
num_features = data[features].select_dtypes(include=['int64', 'float64']).columns
cat_features = data[features].select_dtypes(include=['object']).columns

# Create transformers
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

# Step 2: Feature Engineering (if any)

# Define X and y
X = data[features]
y = data[target]

# Step 3: Model Selection and Training

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to try
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(),
    'GradientBoosting': GradientBoostingRegressor()
}

# Define parameter grids
param_grids = {
    'Ridge': {
        'regressor__alpha': [0.1, 1.0, 10.0]
    },
    'Lasso': {
        'regressor__alpha': [0.1, 1.0, 10.0]
    },
    'RandomForest': {
        'regressor__n_estimators': [100, 200],
        'regressor__max_features': ['auto', 'sqrt'],
        'regressor__max_depth': [10, 20]
    },
    'GradientBoosting': {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.1, 0.05],
        'regressor__max_depth': [3, 4, 5]
    }
}

# Evaluate models
best_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_models[name] = grid_search.best_estimator_
    
    y_pred = grid_search.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"{name} - MAE: {mae}, RMSE: {rmse}, R^2: {r2}")

# Final evaluation of the best model
best_model_name = min(best_models, key=lambda name: mean_squared_error(y_test, best_models[name].predict(X_test)))
best_model = best_models[best_model_name]

print(f"\nBest overall model: {best_model_name}")
y_pred_best = best_model.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"Best Model - MAE: {mae_best}, RMSE: {rmse_best}, R^2: {r2_best}")
