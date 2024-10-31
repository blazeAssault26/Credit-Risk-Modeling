"""
credit_risk.py

This script processes customer data for a credit risk modeling project and 
performs classification using Decision Trees, Random Forest, and XGBoost classifiers.
It includes statistical tests like Chi-Squared and ANOVA for feature selection and 
performs hyperparameter tuning for XGBoost using GridSearchCV.
"""

# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import os

# Loading data from Excel files
a1 = pd.read_excel("/content/drive/MyDrive/bank_internal.xlsx")
a2 = pd.read_excel("/content/drive/MyDrive/cibil.xlsx")

a2.info()

df2 =a2.copy()

import missingno as msno

columns_having_missing_values = (
    df2
    .replace(-99999, np.nan)
    .isna()
    .mean()
    .loc[lambda ser: ser > 0]
    .mul(100)
    .sort_values(ascending=False)
    .index
    .tolist()
)


msno.matrix(df=df2.loc[:, columns_having_missing_values].replace(-99999, np.nan), figsize=(10, 6))

# Handling missing values and removing columns with too many missing values
columns_to_be_removed = df2.columns[(df2 == -99999).sum() > 10000]

df2 = df2.drop(columns=columns_to_be_removed, axis=1)
df2 = df2[df2 != -99999].dropna()

df2.info()

# Merging datasets on 'PROSPECTID'
df = pd. merge ( a1, df2, how ='inner', left_on = ['PROSPECTID'], right_on = ['PROSPECTID'] )

# Chi-Squared Test for Categorical Variables
categorical = df.loc[:, df.columns != 'Approved_Flag'].select_dtypes('object').columns

def perform_chi_squared(column):
    contingency_table = pd.crosstab(df[column], df['Approved_Flag'])
    chi2, pval, _, _ = chi2_contingency(contingency_table)
    return pval

p_values = df[categorical].apply(lambda col: perform_chi_squared(col.name))

results_df = pd.DataFrame({
    'Variable': p_values.index,
    'p-value': p_values.values
})

print("\nChi-Squared Test Results:")
print(results_df)


# Filtering columns based on VIF (Variance Inflation Factor)
numeric_columns = []

for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
        numeric_columns.append(i)
        vif_data = df[numeric_columns]

total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0


for i in range (0,total_columns):

    vif_value = variance_inflation_factor(vif_data, column_index)
    print (column_index,'---',vif_value)


    if vif_value <= 6:
        columns_to_be_kept.append( numeric_columns[i] )
        column_index = column_index+1

    else:
        vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)

print(f"Columns to be kept after calculating vif: {columns_to_be_kept}")


# ANOVA Test for Numerical Variables
grouped_data = df.groupby('Approved_Flag')[columns_to_be_kept]
transposed_grouped_data = [group for name, group in grouped_data]

columns_to_be_kept_numerical = []

for col in columns_to_be_kept:
  group_P1 = grouped_data.get_group('P1')[col]
  group_P2 = grouped_data.get_group('P2')[col]
  group_P3 = grouped_data.get_group('P3')[col]
  group_P4 = grouped_data.get_group('P4')[col]

  f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)
  print(f"Column: {col}, P-Value: {p_value}")

  if p_value <= 0.05:
    columns_to_be_kept_numerical.append(col)

print(f"Columns to be kept after anova test:{columns_to_be_kept_numerical}")

features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

#Encoding categorical variables
education_mapping = {
    'SSC': 1,
    '12TH': 2,
    'GRADUATE': 3,
    'UNDER GRADUATE': 3,
    'POST-GRADUATE': 4,
    'OTHERS': 1,
    'PROFESSIONAL': 3
}

df['EDUCATION'] = df['EDUCATION'].replace(education_mapping)

print(df.head())

df['EDUCATION'].value_counts()
#df['EDUCATION'] = df['EDUCATION'].astype(int)
#df.info()

df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])

y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Model evaluation function
def evaluate_classifier(model, x_train, y_train, x_test, y_test, model_name="Model"):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)


    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

    print(f"\nAccuracy for {model_name}: {accuracy:.2f}")
    for i, v in enumerate(['P1', 'P2', 'P3', 'P4']):
        print(f"Class {v}:")
        print(f"  Precision: {precision[i]:.2f}")
        print(f"  Recall: {recall[i]:.2f}")
        print(f"  F1 Score: {f1_score[i]:.2f}")
        print()

dt_classifier = DecisionTreeClassifier(max_depth=20, min_samples_split=10, random_state=42)
evaluate_classifier(dt_classifier, x_train, y_train, x_test, y_test, model_name="Decision Tree")

rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
evaluate_classifier(rf_classifier, x_train, y_train, x_test, y_test, model_name="Random Forest")

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4, random_state=42)
evaluate_classifier(xgb_classifier, x_train, y_train, x_test, y_test, model_name="XGBoost")

# Hyperparameter tuning for XGBoost using GridSearchCV
def tune_xgboost(x_train, y_train, x_test, y_test):

    param_grid = {
        'learning_rate': [0.1, 0.01,1],
        'reg_lambda' : [0,1,10],
        'max_depth': [6, 7, 5],
        'gamma' : [0.25, 1, 2],
        'n_estimators'    : [200, 300, 500],
        'alpha'           : [1, 10, 100]
    }

    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)


    print("Best Hyperparameters:", grid_search.best_params_)


    best_model = grid_search.best_estimator_
    accuracy = best_model.score(x_test, y_test)
    print("Test Accuracy with Best Hyperparameters:", accuracy)

tune_xgboost(x_train, y_train, x_test, y_test)

# Evaluate the best model from hyperparameter tuning
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',
                                   alpha= 1, learning_rate= 0.1,
                                   max_depth= 5, n_estimators= 200, reg_lambda= 1,  gamma= 2, num_class=4 )
evaluate_classifier(xgb_classifier, x_train, y_train, x_test, y_test, model_name="XGBoost")


# Feature importance and visualization

bst = xgb_classifier.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
  print('%s: ' %importance_type, bst.get_score(importance_type=importance_type))

node_params = {'shape' : 'box', 'style' : 'filled, rounded', 'fillcolor': ' #78cbe ' }
leaf_params  ={'shape' : 'box', 'style' : 'filled', 'fillcolor': ' #e48038 '}

graph_data = xgb.to_graphviz(xgb_classifier, num_trees = 0 , size = "10,10", condition_node_params = node_params, leaf_node_params = leaf_params)
graph_data.view(filename = 'credit_risk_modeling')
