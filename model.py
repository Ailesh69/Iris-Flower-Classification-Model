import pandas as pd 
import joblib
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder , StandardScaler
import warnings 
from sklearn.svm import SVC

iris = pd.read_csv('iris.csv')
print(iris.head())  

label_encoder = LabelEncoder()
iris["species_encoded"] = label_encoder.fit_transform(iris['species'])

print("Encoded Species:")
print(label_encoder.classes_)# to check what species are encoded 
print(iris.groupby('species').mean())# to check the mean of each feature for each species

#visualization
sns.scatterplot(x="sepal_length",y="sepal_width",hue="species",data=iris)
plt.title("Sepal Length vs Sepal Width")
plt.show()

#line plot
sns.lineplot(data = iris.drop(['species_encoded','species'],axis=1))
plt.title("Line Plot of Iris Dataset")
plt.show()

x = iris.drop(['species','species_encoded'], axis=1)
y = iris['species_encoded']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .4,random_state = 42,stratify=y)

svm = SVC(kernel='rbf' , gamma = 0.1 , C =1.0)
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)

print("\nSvm Classification Report")
print(metrics.classification_report(y_test,y_pred))
print(f"\nSvm Accuracy\n{metrics.accuracy_score(y_test,y_pred):.4f}")

joblib.dump(svm , "model.pkl")
print("\nModel saved successfully as model.pkl")