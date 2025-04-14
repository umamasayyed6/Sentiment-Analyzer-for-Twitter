from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Example data
X_train = ["I love this!", "I hate this!"]
y_train = [1, 0]

# Train vectorizer and model
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
