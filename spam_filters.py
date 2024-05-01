import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  #BART
import matplotlib.pyplot as plt

data = pd.read_csv('emails.csv')
X = data['text'] 
y = data['spam']

# Test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorization
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Spam filter models
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(),
    'k-NN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'RandomForest(BART)': RandomForestClassifier()
}

model_evaluation = {
    'Classifier': [],
    'Accuracy': [],
    'Recall': [],
    'Precision': [],
    'F1 Score': []
}

# Train and evaluate the filters
for name, clf in classifiers.items():
    clf.fit(X_train_transformed, y_train)
    y_pred = clf.predict(X_test_transformed)
    model_evaluation['Classifier'].append(name)
    model_evaluation['Accuracy'].append(accuracy_score(y_test, y_pred))
    model_evaluation['Recall'].append(recall_score(y_test, y_pred))
    model_evaluation['Precision'].append(precision_score(y_test, y_pred))
    model_evaluation['F1 Score'].append(f1_score(y_test, y_pred))

results_df = pd.DataFrame(model_evaluation)

# Plotting
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Performance Comparison of Spam filters')

# Accuracy
ax[0, 0].bar(results_df['Classifier'], results_df['Accuracy'], color='blue')
ax[0, 0].set_title('Accuracy')
ax[0, 0].set_ylim([0, 1])
ax[0, 0].set_ylabel('Accuracy')

# Recall
ax[0, 1].bar(results_df['Classifier'], results_df['Recall'], color='green')
ax[0, 1].set_title('Recall')
ax[0, 1].set_ylim([0, 1])
ax[0, 1].set_ylabel('Recall')

# Precision
ax[1, 0].bar(results_df['Classifier'], results_df['Precision'], color='red')
ax[1, 0].set_title('Precision')
ax[1, 0].set_ylim([0, 1])
ax[1, 0].set_ylabel('Precision')

# F1 Score
ax[1, 1].bar(results_df['Classifier'], results_df['F1 Score'], color='purple')
ax[1, 1].set_title('F1 Score')
ax[1, 1].set_ylim([0, 1])
ax[1, 1].set_ylabel('F1 Score')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#end of code
