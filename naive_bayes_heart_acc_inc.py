# The following code has been adapted from https://www.analyticsvidhya.com/blog/2021/01/a-guide-to-the-naive-bayes-algorithm/ 

# Increasing accuracy
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('heart.csv')
x = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7]].values 
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test) #put the false heart disease data here instead of x_test

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
print("Accuracy of heart prediction using naive bayes algorithm:", ac)
cm = confusion_matrix(y_test, y_pred)
print("The confusion matrix is: ", cm)

for i in range (10):
    with open(f'output_simple_gan{i}.npy', 'rb') as fake_heart_disease:
        a = np.load(fake_heart_disease).reshape(32, 7)
        a[:,0] = a[:,0] * 100
        a[:,3] = a[:,3] * 300
        a[:,4] = a[:,4] * 300

    y_pred1 = classifier.predict(a)
print("The accuracy of heart prediction by Naive Bayes algorithm using the dataset obtained from the output of a simple gan, used here an an input is: ", np.sum(y_pred1)/len(y_pred1))
