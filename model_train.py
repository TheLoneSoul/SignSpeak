import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#Load dataset
data_dict = pickle.load(open('./dataset.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

#Splits into training and testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#Train model using RandomForest
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

#Evaluate accuracy
score = accuracy_score(y_prediction, y_test)
print('{}% of image samples were classified Successfully !'.format(score * 100))