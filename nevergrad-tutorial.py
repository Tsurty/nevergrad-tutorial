#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Imports
import pandas as pd
import numpy as np
import plotly.express as px

from numpy import mean, std, absolute

from sklearn import metrics
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, check_array

import nevergrad as ng
import statsmodels.tsa.api as tsa


# In[5]:


df= pd.read_csv('dataset.csv')
df.head()


# In[6]:


# Automatically get the Media Variables
spend_var = [col for col in df.columns if 'spend' in col]
y = df['revenue']

# Print all variables in a Timeseries plot
fig = px.line(df, x='days', y=["revenue", "facebook_spend", "google_spend"])

fig.show()


# In[7]:


# Create and split data variables
X = df[spend_var]
y = df['revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[8]:


# Function to return Adstocked variables
def adstock(x, theta):
    return tsa.filters.recursive_filter(x, theta)

# Function to return Saturated variables
def saturation(x, beta):
    return x ** beta

# Function to return model's NRMSE
def nrmse(y_actual, y_pred):
    # normalized root mean square error
    value = round(np.sqrt(metrics.mean_squared_error(y_actual, y_pred)) / np.mean(y_actual), 3)
    passed = "✔️" if value < 0.15 else "❌"
    return value#, passed

# Function to return model's MAPE
def mape(y_actual, y_pred):
    # mean absolute percentage error
    value = round(metrics.mean_absolute_error(y_actual, y_pred)/np.mean(y_actual),3)
    passed = "✔️" if value < 0.15 else "❌"
    return value#, passed

# Function to return model's R^2
def rsquared(y_actual, y_pred):
    # r squared
    value = round(metrics.r2_score(y_actual, y_pred), 3)
    passed = "✔️" if value > 0.8 else "❌"
    return value#, passed


# In[9]:


# Create a dictionary to hold transformed columns
new_X = {}

# We define one big function that does all the modeling
# This allows us to put all the Hyperparameters in one place and run Nevergrad once for all of them
def build_model(alpha, facebook_spend_theta, facebook_spend_beta, google_spend_theta, google_spend_beta):
    
    # Transform all media variables and set them in the new Dictionary
    # Adstock first and Saturation second
    new_X["facebook_spend"] = saturation(adstock(df["facebook_spend"], facebook_spend_theta), facebook_spend_beta)
    new_X["google_spend"] = saturation(adstock(df["google_spend"], google_spend_theta), google_spend_beta)
    
    # Cast Dictionary to DataFrame and append the output column
    new_df = pd.DataFrame.from_dict(new_X)
    new_df = new_df.join(df['revenue'])

    # Train test split data
    X = new_df[spend_var]
    y = new_df['revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Define the model
    model = Ridge(alpha=alpha)
    # Fit the model using new (transformed) data
    model.fit(X_train, y_train)

    result = df
    # Predict using test data
    result['prediction'] = model.predict(X)

    # Calculate all model's KPIs
    nrmse_val = nrmse(result['revenue'], result['prediction'])
    mape_val = mape(result['revenue'], result['prediction'])
    rsquared_val =rsquared(result['revenue'], result['prediction'])

    # The return should be a value to minimize
    return mape_val


# In[10]:


# Define the list of hyperparameters to optimize
# List must be the same as the ones in the function's definition, same order recommended too
instrum = ng.p.Instrumentation(
    alpha = ng.p.Scalar(),
    
    facebook_spend_theta = ng.p.Scalar(lower=0, upper=1),
    facebook_spend_beta = ng.p.Scalar(lower=0, upper=1),
    
    google_spend_theta = ng.p.Scalar(lower=0, upper=1),
    google_spend_beta = ng.p.Scalar(lower=0, upper=1)
)
# Define an Optimizer (use NGOpt as default) and set budget as number of trials (recommended 2500+)
optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=2500)

# Pass the function to minimize
# Nevergrad will automatically map Hyperparams
recommendation = optimizer.minimize(build_model)


# In[11]:


# Input Nevergrad's recommended values to create the optimised model
model_mape = build_model(**recommendation.value[1]) 

print('MAPE: ', model_mape)


# In[12]:


# Once the minimized variable is good define the function again with more outputs to return
# Rebuild the model with recommended values and use it as you wish
new_X = {}

def build_model(alpha, facebook_spend_theta, facebook_spend_beta, google_spend_theta, google_spend_beta):
    new_X["facebook_spend"] = saturation(adstock(df["facebook_spend"], facebook_spend_theta), facebook_spend_beta)
    new_X["google_spend"] = saturation(adstock(df["google_spend"], google_spend_theta), google_spend_beta)
    
    new_df = pd.DataFrame.from_dict(new_X)
    new_df = new_df.join(df['revenue'])

    X = new_df[spend_var]
    y = new_df['revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    result = df
    result['prediction'] = model.predict(X)

    nrmse_val = nrmse(result['revenue'], result['prediction'])
    mape_val = mape(result['revenue'], result['prediction'])
    rsquared_val =rsquared(result['revenue'], result['prediction'])

    return  mape_val, nrmse_val, rsquared_val, model, result

model_mape, model_nrmse,model_rsq, model, result = build_model(**recommendation.value[1])

# Compare actual vs. predicted values
fig = px.line(result, x='days', y=["revenue", 'prediction'])

fig.show()

print('R^2: ', model_rsq)
print('MAPE: ', model_mape)
print('NRMS: ', model_nrmse)

