
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("data/flights.csv")
data.dropna(inplace=True)

data['Delayed'] = np.where(data['ArrivalDelay'] > 15, 1, 0)

features = ['DayOfWeek', 'DepartureTime', 'Distance', 'Airline']
X = data[features]
y = data['Delayed']

encoder = LabelEncoder()
X['Airline'] = encoder.fit_transform(X['Airline'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()
