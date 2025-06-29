import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Create a PorterStemmer object
ps = PorterStemmer()

# Load the vectorizer and model from pickle files
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Function to transform text
def transform_text(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

# App layout
st.title("Identify spam messages with a click of a button")
st.markdown('Protect yourself from *getting spammed* by using this service.')

# Input message area
st.header("SMS Spam Detection")
input_msg = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    # Preprocess input message
    transformed_msg = transform_text(input_msg)
    # Vectorize input message
    vector_input = tfidf.transform([transformed_msg])
    # Predict using the model
    result = model.predict(vector_input)[0]
    ham, spam = model.predict_proba(vector_input)[0]
    # Display result
    if result == 1:
        with st.spinner('Wait for it...'):
            time.sleep(1)
        st.error("This is a Spam message")
        st.write(f"Spam Probability: {spam * 100:.2f}, Ham Probability: {ham * 100:.2f}")
    else:
        with st.spinner('Wait for it...'):
            time.sleep(1)
        st.success("This is not a Spam Message")
        st.write(f"Spam Probability: {spam * 100:.2f}, Ham Probability: {ham * 100:.2f}")