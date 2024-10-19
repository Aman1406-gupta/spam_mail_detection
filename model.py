import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_emails(folder_path):
    emails = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                emails.append(content)
                labels.append(1 if 'spms' in filename.lower() else 0)
    return emails, labels

train_folder = 'static/train-mails'
test_folder = 'static/test-mails'

X_train, y_train = load_emails(train_folder)
X_test, y_test = load_emails(test_folder)

tfidf = TfidfVectorizer(stop_words='english')

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = []

for model_name, model in models.items():
    
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

print("{:<20} {:<10} {:<10} {:<10} {:<10}".format('Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'))
print("-" * 60)
for result in results:
    print("{:<20} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        result['Model'],
        result['Accuracy'],
        result['Precision'],
        result['Recall'],
        result['F1 Score']
    ))
    
best_model = models['Naive Bayes'] 
pickle.dump(best_model, open('spam_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))