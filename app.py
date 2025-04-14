# app.py
import joblib

# import streamlit as st

# from streamlit_jupyter import StreamlitPatcher, tqdm

# StreamlitPatcher().jupyter()  # register streamlit with jupyter-compatible wrappers
# import streamlit as st
# from sentiment_analysis import analyze_sentiment

# Streamlit front-end
# st.title("Twitter Sentiment Analysis")

# st.write("Analyze the sentiment of a tweet:")

# User input area for entering tweet text
# tweet = st.text_area("Enter a tweet", placeholder="Type a tweet here...")

# Button to submit tweet and analyze
# if st.button("Analyze"):
#     if tweet:
#         sentiment = analyze_sentiment(tweet)
#         st.wvectorizer.pklrite(f"Sentiment: *{sentiment.capitalize()}*")
#     else:
#         st.error("Please enter a tweet.")
        
    #   2  
        
import pickle       
import streamlit as st
import pandas as pd
import os

# Title and Description
st.title("Twitter Sentiment Analysis App")
st.write("""
This application analyzes the sentiment of tweets using a machine learning model trained on the Sentiment140 dataset.
Upload a CSV file of tweets or input your own text to get the sentiment prediction.
""")

# # Sidebar
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose an option:", ["Predict Sentiment", "About"])

# # Load Model and Vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')  # Replace with your model file
        vectorizer = joblib.load('vectorizer.pkl')  # Replace with your vectorizer file
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer files not found. Ensure they are in the app directory.")
        return None, None

model, vectorizer = load_model()

if choice == "Predict Sentiment":
    st.subheader("Sentiment Prediction")
    

#     # Option to input text
    input_type = st.radio("Choose input type:", ["Single Text Input", "Upload CSV"])

    if input_type == "Single Text Input":
        user_input = st.text_area("Enter your tweet text here:")
        if st.button("Predict"):
            if user_input and model and vectorizer:
                input_vector = vectorizer.transform([user_input])
                prediction = model.predict(input_vector)[0]
                sentiment = "Positive" if prediction == 1 else "Negative"
                st.success(f"Predicted Sentiment: {sentiment}")
            else:
                st.warning("Enter text or check model availability.")

    elif input_type == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            if 'text' in data.columns:
                st.write("Data preview:", data.head())
                if st.button("Predict Sentiments"):
                    if model and vectorizer:
                        input_vectors = vectorizer.transform(data['text'])
                        predictions = model.predict(input_vectors)
                        data['sentiment'] = ["Positive" if p == 1 else "Negative" for p in predictions]
                        st.write("Prediction Results:", data)
                        st.download_button("Download Results", data.to_csv(index=False), file_name="results.csv")
            else:
                st.error("The CSV file must have a 'text' column.")
elif choice == "About":
    st.subheader("About this App")
    st.write("""
This app uses a machine learning model trained on the Sentiment140 dataset to analyze sentiment in tweets. 
You can input individual tweets or upload a dataset for bulk analysis.
    """)


joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')


model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
print("Files loaded successfully!")


model = joblib.load('C:\\Users\\91932\\OneDrive\\Desktop\\sentiment analyzer for twitter\\model.pkl')
model = joblib.load('C:\\Users\\91932\\OneDrive\\Desktop\\sentiment analyzer for twitter\\vectorizer.pkl')



# 3

# import streamlit as st
# import pandas as pd
# import joblib
# import os

# # Title and Description
# st.title("Twitter Sentiment Analysis App")
# st.write("""
# This application analyzes the sentiment of tweets using a machine learning model trained on the Sentiment140 dataset.
# You can input a single tweet or upload a CSV file for batch sentiment analysis.
# """)

# Sidebar Navigation
# st.sidebar.title("Navigation")
# choice = st.sidebar.radio("Choose an option:", ["Predict Sentiment", "About"])

# Load Model and Vectorizer
# @st.cache_resource
# def load_model():
#     try:
#         model = joblib.load('model.pkl')
#         vectorizer = joblib.load('vectorizer.pkl')
#         return model, vectorizer
#     except FileNotFoundError:
#         st.error("Model or vectorizer files not found. Ensure they are in the app directory.")
#         return None, None

# model, vectorizer = load_model()

# if choice == "Predict Sentiment":
#     st.subheader("Sentiment Prediction")
#     input_type = st.radio("Choose input type:", ["Single Text Input", "Upload CSV"])

#     if input_type == "Single Text Input":
#         user_input = st.text_area("Enter your tweet text here:")
#         if st.button("Predict"):
#             if user_input and model and vectorizer:
#                 input_vector = vectorizer.transform([user_input])
#                 prediction = model.predict(input_vector)[0]
#                 sentiment = "Positive" if prediction == 1 else "Negative"
#                 st.success(f"Predicted Sentiment: {sentiment}")
#             else:
#                 st.warning("Enter text or check model availability.")

#     elif input_type == "Upload CSV":
#         uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type=['csv'])
#         if uploaded_file is not None:
#             data = pd.read_csv(uploaded_file)
#             if 'text' in data.columns:
#                 st.write("Data preview:", data.head())
#                 if st.button("Predict Sentiments"):
#                     if model and vectorizer:
#                         input_vectors = vectorizer.transform(data['text'])
#                         predictions = model.predict(input_vectors)
#                         data['sentiment'] = ["Positive" if p == 1 else "Negative" for p in predictions]
#                         st.write("Prediction Results:", data)
#                         st.download_button("Download Results", data.to_csv(index=False), file_name="results.csv")
#             else:
#                 st.error("The CSV file must have a 'text' column.")
# elif choice == "About":
#     st.subheader("About this App")
#     st.write("""
# This app uses a machine learning model trained on the Sentiment140 dataset to analyze sentiment in tweets. 
# You can input individual tweets or upload a dataset for bulk analysis.
# """)






















































# /path/to/your/vectorizer.pkl

