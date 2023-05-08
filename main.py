# Imports
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('KSI.csv')

# Describe the data elements (columns) in the DataFrame
print(df.info())

# Describe the count, mean, standard deviation, minimum and maximum values for each column
print(df.describe())

# Provide descriptions and types
print(df.dtypes)

# Print out the unique values and their frequencies for each column
for column in df.columns:
    print(f"Column {column}:")
    print(df[column].value_counts())
    print()

# Show the range of each numeric column in the DataFrame
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        print(f"Range of {column}: {df[column].min()} to {df[column].max()}")

# Show the unique values and their frequencies for each non-numeric column in the DataFrame
for column in df.columns:
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Unique values for {column}:")
        print(df[column].value_counts())
        print()

# Calculate the mean, median, and mode for each numeric column in the DataFrame
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        print(f"Statistics for {column}:")
        print(f"Mean: {df[column].mean()}")
        print(f"Median: {df[column].median()}")
        print(f"Mode: {df[column].mode().values}")
        print()

# Calculate the correlation coefficients between all pairs of numeric columns in the DataFrame
corr_matrix = df.corr(numeric_only=True)
print("Correlation matrix:")
print(corr_matrix)

# Check if there are any missing values in the DataFrame
if df.isnull().values.any():
    print("There are missing values in the DataFrame")
else:
    print("There are no missing values in the DataFrame")

# Count the number of missing values in each column of the DataFrame
print("Number of missing values in each column:")
print(df.isnull().sum())

# Convert the date column to a datetime object
df['DATE'] = pd.to_datetime(df['DATE'])

# Extract the year from the date column
df['YEAR'] = df['DATE'].dt.year

# Count the number of collisions per year
year_counts = df['YEAR'].value_counts()

# Plot a bar chart of the collision counts per year
plt.bar(year_counts.index, year_counts.values)
plt.title('Number of Collisions per Year')
plt.xlabel('Year')
plt.ylabel('Number of Collisions')
plt.show()

# Scatter plot of the longitude and latitude columns
plt.scatter(df['LONGITUDE'], df['LATITUDE'], s=1, alpha=0.1)
plt.title("Toronto Traffic Accidents")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Drop the rows that have Property Damage Only has there's only 7 occurences and we are only looking for the outcome of fatal or non-fatal
df = df.drop(df[df['ACCLASS'] == 'PROPERTY DAMAGE ONLY'].index, axis=0)

# Drop irrelevant columns
df.drop(
    ['INDEX_', 'ACCNUM', 'YEAR', 'DATE', 'INVAGE', 'INJURY', 'FATAL_NO', 'HOOD_158', 'NEIGHBOURHOOD_158', 'HOOD_140',
     'NEIGHBOURHOOD_140', 'DIVISION', 'ObjectId'], axis=1, inplace=True)

# Select relevant features
features = ['TIME', 'STREET1', 'STREET2', 'OFFSET', 'ROAD_CLASS', 'DISTRICT', 'WARDNUM', 'LATITUDE', 'LONGITUDE',
            'LOCCOORD', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'INITDIR',
            'VEHTYPE', 'MANOEUVER', 'DRIVACT', 'DRIVCOND', 'PEDTYPE', 'PEDACT', 'PEDCOND', 'CYCLISTYPE', 'CYCACT',
            'CYCCOND', 'PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'EMERG_VEH',
            'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']
target = 'ACCLASS'

# Create a new DataFrame with only the selected features and target
df_new = df[features + [target]]

# Define the features and target
X = df_new[features]
y = df_new[target]

# Separate majority and minority classes
df_majority = df_new[df_new[target] == 'Non-Fatal Injury']
df_minority = df_new[df_new[target] == 'Fatal']

# Upsample the minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=len(df_majority),  # to match majority class
                                 random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# create a LabelEncoder object
le = LabelEncoder()

# fit and transform the ACCLASS column
df_balanced['ACCLASS'] = le.fit_transform(df_balanced['ACCLASS'])

# Define the features and target for the balanced dataset
oversample = RandomOverSampler(sampling_strategy='minority')
X_balanced_old = df_balanced[features]
y_balanced_old = df_balanced['ACCLASS']

X_balanced, y_balanced = oversample.fit_resample(X_balanced_old, y_balanced_old)

# Define the preprocessing steps for the numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define the preprocessing steps for the categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Use ColumnTransformer to apply the preprocessing steps to the appropriate columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X_balanced.select_dtypes(include=['float64', 'int64']).columns),
        ('cat', categorical_transformer, X_balanced.select_dtypes(include=['object']).columns)
    ])

# Define the logistic regression pipeline
pipeline_logistic = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())

])

# Save logistic regression pipeline
joblib.dump(pipeline_logistic, 'pipeline_logistic.pkl')

# Define the decision tree classifier pipeline
pipeline_decision = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Save decision tree pipeline
joblib.dump(pipeline_decision, 'pipeline_decision.pkl')

# Define the SVM classifier
pipeline_SVM = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm.SVC())
])

# Save neural pipeline
joblib.dump(pipeline_SVM, 'pipeline_SVM.pkl')

# Define the random forest classifier
pipeline_forest = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Save random forest pipeline
joblib.dump(pipeline_forest, 'pipeline_forest.pkl')

# Define the neural network classifier
pipeline_neural = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier())
])

# Save neural pipeline
joblib.dump(pipeline_neural, 'pipeline_neural.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Create a CSV file out of my testing data to use later in testing my model API service
test_df = pd.DataFrame(X_test, columns=X_balanced.columns)

test_df.to_csv('test_data.csv', index=False)

################ DECISION TREE ################

# Parameter grid for decision trees
param_grid_decision = {
    'classifier__max_depth': [2, 4, 6, 8, 10, None],
    'classifier__min_samples_leaf': [1, 2, 4, 8, 16],
    'classifier__min_samples_split': [2, 5, 10, 20],
    'classifier__max_features': ['sqrt', 'log2', None]
}

# Perform randomized grid search to find the best hyperparameters for the model
random_search_decision = RandomizedSearchCV(pipeline_decision, param_distributions=param_grid_decision,
                                            n_iter=50, cv=5, scoring='roc_auc', random_state=42)

# Fit the pipeline with the best hyperparameters found during the randomized grid search
random_search_decision.fit(X_train, y_train)

# Make predictions on the test set
y_pred_decision = random_search_decision.predict(X_test)

# Calculate evaluation metrics
accuracy_decision = accuracy_score(y_test, y_pred_decision)
precision_decision = precision_score(y_test, y_pred_decision)
recall_decision = recall_score(y_test, y_pred_decision)
f1_decision = f1_score(y_test, y_pred_decision)
conf_matrix_decision = confusion_matrix(y_test, y_pred_decision)

# Print the evaluation metrics and confusion matrix
print("DECISION TREES")

print('Accuracy: {:.2f}'.format(accuracy_decision))
print('Precision: {:.2f}'.format(precision_decision))
print('Recall: {:.2f}'.format(recall_decision))
print('F1 score: {:.2f}'.format(f1_decision))
print('Confusion matrix:\n', conf_matrix_decision)

# Calculate and plot ROC curve
y_score_decision = random_search_decision.predict_proba(X_test)[:, 1]
fpr, tpr, thresholdss = roc_curve(y_test, y_score_decision)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc='lower right')
plt.show()

# Best model for decision trees
best_model_decision = random_search_decision.best_estimator_

# Save the best neural model using joblib
joblib.dump(best_model_decision, 'best_model_decision.pkl')

################ LOGISTIC REGRESSION (DID NOT CONVERGE) ################

# Define the hyperparameter grid for the Logistic Regression model
param_grid_logistic = {
    'classifier__penalty': ['l1', 'l2'],
    'classifier__C': np.logspace(-4, 4, 100),
    'classifier__solver': ['liblinear', 'saga', 'newton-cg', 'lbfgs', 'sag'],
    'classifier__max_iter': np.arange(100, 1000, 100),
    'classifier__tol': np.logspace(-6, -2, 10),
}

# Perform randomized grid search to find the best hyperparameters for the model
random_search_logistic = RandomizedSearchCV(pipeline_logistic, param_distributions=param_grid_logistic,
                                            n_iter=50, cv=5, scoring='roc_auc', random_state=42)

# Fit the pipeline with the best hyperparameters found during the randomized grid search (DID NOT CONVERGE)
random_search_logistic.fit(X_train, y_train)

# Make predictions on the test set
y_pred = random_search_logistic.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# Print the evaluation metrics and confusion matrix
print('Accuracy: {:.2f}'.format(accuracy))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 score: {:.2f}'.format(f1))
print('Confusion matrix:\n', conf_matrix)

# Calculate and plot ROC curve
y_score = random_search_logistic.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')# plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc='lower right')
plt.show()

# Best model for neural networks
best_model_logistic = random_search_logistic.best_estimator_

# Save the best neural model using joblib
joblib.dump(best_model_logistic, 'best_model_logistic.pkl')


################ SVM ################

# Parameter grid for svm
param_grid_svm = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'classifier__degree': [2, 3, 4],
    'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 6))
}

# Perform randomized grid search to find the best hyperparameters for the model
random_search_svm = RandomizedSearchCV(pipeline_SVM, param_distributions=param_grid_svm,
                                       n_iter=50, cv=5, scoring='roc_auc', random_state=42)

# Fit the pipeline with the best hyperparameters found during the randomized grid search
random_search_svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = random_search_svm.predict(X_test)

# Calculate evaluation metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Print the evaluation metrics and confusion matrix
print("SVM")

print('Accuracy: {:.2f}'.format(accuracy_svm))
print('Precision: {:.2f}'.format(precision_svm))
print('Recall: {:.2f}'.format(recall_svm))
print('F1 score: {:.2f}'.format(f1_svm))
print('Confusion matrix:\n', conf_matrix_svm)

# Calculate and plot ROC curve
y_score_svm = random_search_svm.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, y_score_svm)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc='lower right')
plt.show()

# Best model for svm
best_model_svm = random_search_svm.best_estimator_

# Save the best neural model using joblib
joblib.dump(best_model_svm, 'best_model_svm.pkl')

################ Random Forest ################

# Parameter grid for random forest
param_grid_forest = {
    'classifier__n_estimators': [100, 300, 500, 800],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth': [None, 5, 10, 20, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Perform randomized grid search to find the best hyperparameters for the model
random_search_forest = RandomizedSearchCV(pipeline_forest, param_distributions=param_grid_forest,
                                          n_iter=50, cv=5, scoring='roc_auc', random_state=42)

# Fit the pipeline with the best hyperparameters found during the randomized grid search
random_search_forest.fit(X_train, y_train)

# Make predictions on the test set
y_pred_forest = random_search_forest.predict(X_test)

# Calculate evaluation metrics
accuracy_forest = accuracy_score(y_test, y_pred_forest)
precision_forest = precision_score(y_test, y_pred_forest)
recall_forest = recall_score(y_test, y_pred_forest)
f1_forest = f1_score(y_test, y_pred_forest)
conf_matrix_forest = confusion_matrix(y_test, y_pred_forest)

# Print the evaluation metrics and confusion matrix
print("RANDOM FOREST")

print('Accuracy: {:.2f}'.format(accuracy_forest))
print('Precision: {:.2f}'.format(precision_forest))
print('Recall: {:.2f}'.format(recall_forest))
print('F1 score: {:.2f}'.format(f1_forest))
print('Confusion matrix:\n', conf_matrix_forest)

# Calculate and plot ROC curve
y_score_forest = random_search_forest.predict_proba(X_test)[:, 1]
fpr, tpr, thresholdsss = roc_curve(y_test, y_score_forest)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc='lower right')
plt.show()

# Best model for neural networks
best_model_forest = random_search_forest.best_estimator_

# Save the best neural model using joblib
joblib.dump(best_model_forest, 'best_model_forest.pkl')

################ Neural Networks ################

# Parameter grid for neural networks
param_grid_neural = {
    'classifier__hidden_layer_sizes': [(50,), (100,), (150,), (200,), (250,)],
    'classifier__activation': ['relu', 'logistic'],
    'classifier__solver': ['adam', 'sgd'],
    'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
    'classifier__learning_rate': ['constant', 'adaptive'],
}

# Perform randomized grid search to find the best hyperparameters for the model
random_search_neural = RandomizedSearchCV(pipeline_neural, param_distributions=param_grid_neural,
                                          n_iter=50, cv=5, scoring='roc_auc', random_state=42)

# Fit the pipeline with the best hyperparameters found during the randomized grid search
random_search_neural.fit(X_train, y_train)

# Make predictions on the test set
y_pred_neural = random_search_neural.predict(X_test)

# Calculate evaluation metrics
accuracy_neural = accuracy_score(y_test, y_pred_neural)
precision_neural = precision_score(y_test, y_pred_neural)
recall_neural = recall_score(y_test, y_pred_neural)
f1_neural = f1_score(y_test, y_pred_neural)
conf_matrix_neural = confusion_matrix(y_test, y_pred_neural)

# Print the evaluation metrics and confusion matrix
print("NEURAL NETWORKS")

print('Accuracy: {:.2f}'.format(accuracy_neural))
print('Precision: {:.2f}'.format(precision_neural))
print('Recall: {:.2f}'.format(recall_neural))
print('F1 score: {:.2f}'.format(f1_neural))
print('Confusion matrix:\n', conf_matrix_neural)

# Calculate and plot ROC curve
y_score_neural = random_search_neural.predict_proba(X_test)[:, 1]
fpr, tpr, thresholdssss = roc_curve(y_test, y_score_neural)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc='lower right')
plt.show()

# Best model for neural networks
best_model_neural = random_search_neural.best_estimator_

# Save the best neural model using joblib
joblib.dump(best_model_neural, 'best_model_neural.pkl')
