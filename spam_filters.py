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

# Plotting the table
plt.figure(figsize=(10, 5))
plt.title('Performance Metrics Comparison')
plt.table(cellText=results_df.values,
          colLabels=results_df.columns,
          loc='center')
plt.axis('off')
plt.show()


plt.figure(figsize=(10, 5))
plt.bar(results_df['Classifier'], results_df['Accuracy'], color='tan')
plt.title('Accuracy Comparison of Spam Filters')
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(results_df['Classifier'], results_df['Recall'], color='cyan')
plt.title('Recall Comparison of Spam Filters')
plt.ylim([0, 1])
plt.ylabel('Recall')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(results_df['Classifier'], results_df['Precision'], color='lightsalmon')
plt.title('Precision Comparison of Spam Filters')
plt.ylim([0, 1])
plt.ylabel('Precision')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(results_df['Classifier'], results_df['F1 Score'], color='lavender')
plt.title('F1-score Comparison of Spam Filters')
plt.ylim([0, 1])
plt.ylabel('F1-score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#end of code