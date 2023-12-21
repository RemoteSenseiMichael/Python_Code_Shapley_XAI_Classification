#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split, GridSearchCV


# In[3]:


from sklearn.ensemble import RandomForestRegressor


# In[4]:


from sklearn.ensemble import RandomForestClassifier


# In[5]:


from sklearn.metrics import mean_squared_error, make_scorer, r2_score, accuracy_score


# In[6]:


from sklearn.model_selection import KFold


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


import shap


# In[9]:


# Import your data containing samples:
data = pd.read_csv('E:/013_Projects/016_ABMI/016_Prototypes/ABMI_Pilots_ML/Boreal_2/temp2/Boreal_2_Samples_NoALOS.csv')


# In[10]:


# Split data into input features and target variable
X = data.drop(columns=['class'])  # Features (input covariates)
y = data['class']  # Target variable


# In[11]:


# GEE Parameters
# n_estimators = numberOfTrees
# max_depth = maxNodes
# min_samples_split = variablesPerSplit
# min_samples_leaf = minLeafPopulation


# In[13]:


param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8]
}


# In[14]:


rf_model = RandomForestClassifier()


# In[15]:


# Step 6: Use GridSearchCV for hyperparameter tuning and 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = make_scorer(accuracy_score)
#grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring=scoring, cv=kf, n_jobs=-1)
grid_search.fit(X, y)


# In[16]:


# Step 7: Extract R² values for each hyperparameter combination
results = pd.DataFrame(grid_search.cv_results_)


# In[17]:


accuracy_values = results[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score']]


# In[18]:


# Step 8: Export R² values to a CSV file
accuracy_values.to_csv('E:/013_Projects/016_ABMI/016_Prototypes/ABMI_Pilots_ML/Boreal_2/temp2/Boreal_2_CV.csv', index=False)


# In[19]:


best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[20]:


print("Best Hyperparameters:", best_params)
print("Best R-squared (R²):", best_accuracy)


# In[21]:


import shap


# In[22]:


# custom_colors = ("blue", "red", "yellow", "orange", "black")  # Set your desired colors here


# In[23]:


explainer = shap.Explainer(best_rf_model, X, check_additivity=False)


# In[ ]:





# In[24]:


shap_values = explainer.shap_values(X, check_additivity=False)


# In[ ]:


# Choosing colors: https://matplotlib.org/stable/users/explain/colors/colormaps.html


# In[25]:


# See the summary plot:
shap.summary_plot(shap_values, X, max_display=20, plot_type="bar", color=plt.get_cmap("tab20c"))


# In[27]:


# Export the summary plot.
plt.figure()
shap.summary_plot(shap_values, X, max_display=20, plot_type="bar", color=plt.get_cmap("tab20c"), show=False)
plt.savefig('E:/013_Projects/016_ABMI/016_Prototypes/ABMI_Pilots_ML/Boreal_2/temp2/Boreal_2_SHAP.jpg', dpi=600)
plt.close()


# In[ ]:




