import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay, precision_score,recall_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


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

print(X_train.head())


model=AdaBoostClassifier()
model.fit(X_train,y_train)

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(y_train,X_train_prediction)
print("Accuracy of training data: ",round(training_data_accuracy*100,2),"%")

y_pred=model.predict(X_test)
print(y_pred)

print("Accuracy of y_pred = ",accuracy_score(y_test,y_pred))

#precision=tp/(tp+fp)=tp/total predicted positive
precision_train=precision_score(y_train,X_train_prediction)
print("Training data precision: ",precision_train)

precision_test=precision_score(y_test,X_train_prediction)
print("Test data precision: ",precision_test)



