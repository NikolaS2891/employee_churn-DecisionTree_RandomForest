import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split

# load data (this is just few rows of random data for example)
data = pd.read_excel('employee.xlsx',sheet_name='Sheet2')
features = list(data.columns[:7])
target = list(data['left'].values)

y = data["left"]
X = data[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

#RandomForestClassifer
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)
y_pred_test = forest.predict(X_test)
print(accuracy_score(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))

#DecisionTreeClassifier
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X,y)

#RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(X, y)

#Plot figure for DecisionTreeClassifier
"""
plt.figure(figsize=(15,8))
a_plot = plot_tree(clf, feature_names=features, class_names=str(target), fontsize=14) #
plt.show()
"""
# Predict on new values
print (clf.predict([[0.55, 0.53, 1, 4, 0, 0, 1]]))
print (clf.predict([[0.75, 0.68, 3, 2, 1, 1, 1]]))
print (clf.predict([[0.72, 0.65, 0, 1, 0, 0, 1]]))