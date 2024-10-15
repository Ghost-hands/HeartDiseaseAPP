import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle


data = pd.read_csv("Heart_Disease_Prediction.csv")
print(data.head())

print(data.isnull().sum())

features = data[["Age", "Chest pain type", "BP", "Cholesterol", "Max HR", "ST depression", "Number of vessels fluro", "Thallium"]]
target = data['Heart Disease']

print(features)
print(target)

x_train, x_test, y_train, y_test = train_test_split(features, target, random_state = 3136)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(x_test)
print(y_test)
print(y_pred)

print("Accuracy score:", accuracy_score(y_pred, y_test))

cr = classification_report(y_pred, y_test)
print("Classification report:")
print(cr)


with open('heartprediction2.pickle', 'wb') as f:
    pickle.dump(model, f)
