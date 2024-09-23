import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Income Prediction using Adaboost Algorithm/adult/adult.data")

# Preprocess the data
df[df == '?'] = np.nan
print(df.isnull().sum())  # Check for missing values

# One-Hot Encoding for categorical variables
categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
df = pd.get_dummies(df, columns=categorical, drop_first=True)

X = df.drop(['income'], axis=1)
y = df['income'].map({' <=50K': 0, ' >50K': 1})  # Binary encoding for income

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Standardizing the numerical data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Train the model
model = AdaBoostClassifier(n_estimators=50, learning_rate=1)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy of y_pred = ", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.title("Confusion Matrix", fontsize=12)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

print("\nConfusion Matrix")
print(classification_report(y_test, y_pred))

# Summary statistics
no_of_records = df.shape[0]
greater_than_50k = df[df['income'] == ' >50K'].shape[0]
less_than_equal_to_50k = df[df['income'] == ' <=50K'].shape[0]
greater_percent = (greater_than_50k / no_of_records) * 100
print("Total number of records: {}".format(no_of_records))
print("Individuals making more than 50K: {}".format(greater_than_50k))
print("Individuals making at most 50K: {}".format(less_than_equal_to_50k))
print("Percentage of individuals making more than 50K: {}%".format(greater_percent))
