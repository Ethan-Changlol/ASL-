import pandas as pd
import string
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('hand_landmarks.csv', header=None)
indices = {}
for c in string.ascii_uppercase:
    indices[c] = df[df[0]==c].index

for i in indices:
    df = df.drop(indices[i][100:])
X = df[list(range(1, 64))]
y = df[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)


dump(clf, 'asl.joblib')