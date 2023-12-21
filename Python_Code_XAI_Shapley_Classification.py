import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import shap

# Import your data as a CSV file containing your samples.
# The first column should be the target variable. The remaining columns should be the predictor variables and their values.
data = pd.read_csv('E:/Samples.csv')

# Split data into input features and target variable which in this case is 'class'.
X = data.drop(columns=['class'])  # Features (input covariates)
y = data['class']  # Target variable

# Here is a comparision list between ee.Classifier.smileRandomForest hyperparameter names and  sklearn Random Forest for reference:
# n_estimators = numberOfTrees
# max_depth = maxNodes
# min_samples_split = variablesPerSplit
# min_samples_leaf = minLeafPopulation

# Set the values for each hyperparameter you would like to optimize (tune) for your Random Forest model.
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create your Random Forest model.
rf_model = RandomForestRegressor(random_state=42)

# Use GridSearchCV for hyperparameter tuning and 5-fold cross-validation. Choose best parameters based on R².
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = make_scorer(accuracy_score)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring=scoring, cv=kf, n_jobs=-1)
grid_search.fit(X, y)

# Extract accuracy values for each hyperparameter combination.
results = pd.DataFrame(grid_search.cv_results_)

# Create accuracy values.
acc_values = results[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score']]

# Export R² values to a CSV file.
acc_values.to_csv('E:/RF_Tuning_Results.csv', index=False)

# Build optimized Random Forest model based on best accuracy.
best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_acc = grid_search.best_score_

# Print the optimized hyperparametes and R² achieved on cross-validation.
print("Best Hyperparameters:", best_params)
print("Best accuracy:", best_acc)

# Use the Shapley Explainer function to analyze your optimzed Random Forest model.
explainer = shap.Explainer(best_rf_model, X, check_additivity=False)

# See (view) the Shap summary plot:
shap_values = explainer.shap_values(X, check_additivity=False)

# Link for choosing bar colors: https://matplotlib.org/stable/users/explain/colors/colormaps.html

# See (view) the summary plot:
shap.summary_plot(shap_values, X, max_display=20, plot_type="bar", color=plt.get_cmap("tab20c"))

# Export the summary plot.
plt.figure()
shap.summary_plot(shap_values, X, max_display=20, plot_type="bar", color=plt.get_cmap("tab20c"), show=False)
plt.savefig('E:/SHAP.jpg', dpi=600)
plt.close()
