import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler

'''STEP 1: data preprocessing
Load the dataset using pandas and explore it for missing values and data types.'''

df=pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Income Prediction using Adaboost Algorithm/adult/adult.data")

#print(df.head())
#  print(df.info())

print(df.shape)
df[df=='?']=np.nan
#print(df.info())

#Encode categorical variables (e.g., using one-hot encoding for education and occupation).

print(df.isnull().sum()) #Since there are no missing values, no need to use mode 

X=df.drop(['income'],axis=1)
y=df['income']

print(X.head())

'''Train the Model:
Split the dataset into training (e.g., 80%) and testing (e.g., 20%) sets.'''
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)

#convert categorical data to numerical data
categorical=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
for feature in categorical:
    label=preprocessing.LabelEncoder()
    X_train[feature]=label.fit_transform(X_train[feature])
    X_test[feature]=label.transform(X_test[feature])

#standardizing the data using StandardScaler()
scaler=StandardScaler()
X_train=pd.DataFrame(scaler.fit_transform(X_train),columns=X.columns)
X_test=pd.DataFrame(scaler.transform(X_test),columns=X.columns)

#print(X_train.head())

model=AdaBoostClassifier(n_estimators=50,learning_rate=1)
model.fit(X_train,y_train)

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(y_train,X_train_prediction)
print("Accuracy of training data: ",round(training_data_accuracy*100,2),"%")

y_pred=model.predict(X_test)
print(y_pred)

print("Accuracy of y_pred = ",accuracy_score(y_test,y_pred))

cm=confusion_matrix(y_test,y_pred)
plt.title("Confusion matrix",fontsize=12)
sns.heatmap(cm,annot=True,fmt='d')
plt.show()

print("\nConfusion Matrix")
print(classification_report(y_test,y_pred))


no_of_records=df.shape[0]
greater_than_50k=df[df['income']==' >50K'].shape[0]
less_than_equal_to_50k=df[df['income']==' <=50K'].shape[0]
greater_percent=(greater_than_50k/no_of_records)*100
print("Total number of records: {}".format(no_of_records))
print("Individuals making more than 50K: {}".format(greater_than_50k))
print("Individuals making at most 50K: {}".format(less_than_equal_to_50k))
print("Percentage of individuals making more than 50K: {}%".format(greater_percent))


