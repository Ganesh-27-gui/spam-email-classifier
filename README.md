Spam Email Classifier using Naive Bayes

import pandas as pd from sklearn.feature_extraction.text import CountVectorizer from sklearn.model_selection import train_test_split from sklearn.naive_bayes import MultinomialNB from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

Step 1: Load the dataset

df = pd.read_csv('spam.csv', encoding='latin-1') df = df[['v1', 'v2']] df.columns = ['label', 'message']

Step 2: Preprocess labels

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

Step 3: Feature extraction using CountVectorizer

vectorizer = CountVectorizer() X = vectorizer.fit_transform(df['message']) y = df['label']

Step 4: Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Step 5: Train Naive Bayes model

model = MultinomialNB() model.fit(X_train, y_train)

Step 6: Make predictions

y_pred = model.predict(X_test)

Step 7: Evaluate model

print("Accuracy:", accuracy_score(y_test, y_pred)) print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) print("Classification Report:\n", classification_report(y_test, y_pred))

.
