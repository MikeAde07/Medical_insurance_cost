import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# load dataset

ins_data = pd.read_csv("insurance.csv")

# investigate the dataset
#print(ins_data.head())
#print(ins_data.tail())
#print(ins_data.describe())
#print(ins_data.info())

#print(ins_data.shape)

#print(ins_data.isnull().sum())

#distribution of age value
#sns.set_theme()
#plt.figure(figsize=(6,6))
#sns.histplot(ins_data["age"])
#plt.title("Age Distribution")
#plt.show()

# Gender column
#plt.figure(figsize=(6,6))
#sns.countplot(x="sex", data = ins_data)
#plt.title("Sex Distribution")
#plt.show()

#print(ins_data["sex"].value_counts())

# BMI distribution 
#sns.set_theme()
#plt.figure(figsize=(6,6))
#sns.histplot(ins_data["bmi"])
#plt.title("BMI Distribution")
#plt.show()

#print(ins_data["bmi"].describe())
#print(ins_data["children"].value_counts())

# children column
#plt.figure(figsize=(6,6))
#sns.countplot(x="children", data = ins_data)
#plt.title("Children")
#plt.show()

# smoker column
#plt.figure(figsize=(6,6))
#sns.countplot(x="smoker", data = ins_data)
#plt.title("Smokers")
#plt.show()

#print(ins_data["smoker"].value_counts())

# region column
#plt.figure(figsize=(6,6))
#sns.countplot(x="region", data = ins_data)
#plt.title("Region")
#plt.show()

#print(ins_data["region"].value_counts())

# distribution of charges
#sns.set_theme()
#plt.figure(figsize=(6,6))
#sns.histplot(ins_data["charges"])
#plt.title("Charge Distribution")
#plt.show()

# Encoding the categorical features

## encoding the sex column
ins_data.replace({"sex": {"male": 0, "female": 1}}, inplace=True)

## encoding the smoker column
ins_data.replace({"smoker": {"yes": 0, "no": 1}}, inplace=True)

## encoding the region column
ins_data.replace({"region": {"southeast": 0, "southwest": 1, "northeast": 2, "northwest": 3}}, inplace=True)

# Split the data into features and target

X = ins_data.drop("charges", axis=1)
#print(X.head())

Y = ins_data["charges"]
#print(Y.tail())

# train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
kf = KFold(n_splits=5, shuffle=True, random_state=3)

#print(X.shape, X_train.shape, X_test.shape)
#print(Y.shape, Y_train.shape, Y_test.shape)

# Model Training & evaluation
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

X_train_pred = regressor.predict(X_train)
# R squared value
r2_train = metrics.r2_score(Y_train, X_train_pred) 
print("R squared value : ", r2_train)

# prediction on test data
Y_pred = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, Y_pred)
cv_results = cross_val_score(regressor, X, Y, cv=kf)
#mean_s2_error = metrics.root_mean_squared_error(Y_test, Y_pred, squared=True)
print("R squared value test data : ", r2_test)
print("Cross Validation : ", cv_results)

# Build a predictive system
input_data = (31, 1, 25.74, 0, 1, 0)
input_data_np = np.asarray(input_data)
input_reshape = input_data_np.reshape(1,-1)
pred = regressor.predict(input_reshape)

print("Charge will be (USD): " , pred[0])