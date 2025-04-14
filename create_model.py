import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data for demonstration (replace with your dataset)
data = {
    "text": [
        "I love this product!", 
        "This is the best purchase I've made.", 
        "Absolutely terrible experience.", 
        "I hate this item!", 
        "It's okay, not great.",
        "I am so happy with this!", 
        "Worst thing ever!", 
        "I enjoy using it.",
        "Awful quality, very disappointed.", 
        "Pretty good, would recommend."
    ],
    "sentiment": [1, 1, 0, 0, 0, 1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
X_test_vectorized = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model and vectorizer to disk
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully!")
