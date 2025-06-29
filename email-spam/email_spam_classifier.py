import pickle
import string
import imaplib
import email
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from functools import lru_cache
import streamlit as st

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Create a PorterStemmer object
ps = PorterStemmer()

# Load the vectorizer and model from pickle files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to transform text
def transform_text(text: str) -> str:
    """Preprocess the input text for prediction."""
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return ' '.join(text)

# Function to check if a label exists in Gmail
@lru_cache(maxsize=3)
def check_label_exists(label_name: str) -> bool:
    """Check if the label exists in Gmail."""
    result, labels = mail.list()
    if result == 'OK':
        for label in labels:
            if label_name.encode() in label:
                return True
    return False

# Gmail credentials (replace with your credentials or use environment variables)
EMAIL = 'dhruvaagarwal90@gmail.com'
PASSWORD = 'dwdr ozpj bofx grdw'

# Connect to Gmail IMAP server
mail = imaplib.IMAP4_SSL('imap.gmail.com')
mail.login(EMAIL, PASSWORD)
mail.select('inbox')

# Search for latest 100 emails in inbox
result, data = mail.search(None, 'ALL')
email_ids = data[0].split()
last_50_emails = email_ids[-50:][::-1]

# Streamlit UI
st.title("Email Spam Detection Results")
st.write("Below are the latest 50 emails and their spam predictions:")

for eid in last_50_emails:
    result, msg_data = mail.fetch(eid, '(RFC822)')
    # Ensure msg_data[0] is a tuple and the second element is bytes
    if (
        isinstance(msg_data, list)
        and len(msg_data) > 0
        and isinstance(msg_data[0], tuple)
        and len(msg_data[0]) > 1
        and isinstance(msg_data[0][1], (bytes, bytearray))
    ):
        raw_email = msg_data[0][1]
        if isinstance(raw_email, bytes) or isinstance(raw_email, bytearray):
            msg = email.message_from_bytes(raw_email)
        else:
            st.warning("Email data is not in bytes format, skipping this email.")
            continue
    else:
        st.warning("Email data is not in the expected format, skipping this email.")
        continue

    subject = msg['subject']
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    body = payload.decode(errors='replace')
                elif isinstance(payload, str):
                    body = payload
                else:
                    body = ""
                break
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            body = payload.decode(errors='replace')
        elif isinstance(payload, str):
            body = payload
        else:
            body = ""

    # Combine subject and body for prediction
    email_text = f"{subject} {body}"

    # Preprocess and vectorize
    transformed = transform_text(email_text)
    vectorized = tfidf.transform([transformed]).toarray()

    # Predict
    prediction = model.predict(vectorized)[0]
    ham, spam = model.predict_proba(vectorized)[0]
    label = 'Spam' if prediction == 1 else 'Ham'

    # Show on UI
    color = "red" if label == "Spam" else "green"
    # Custom HTML/CSS for colored expander header
    expander_html = f"""
        <details style="border: 2px solid {color}; border-radius: 8px; margin-bottom: 10px;">
            <summary style="font-weight: bold; color: {color}; font-size: 18px;">
                Subject: {subject}
            </summary>
            <div style="padding: 10px;">
                <p><b>Body:</b> {body}</p>
                <p><b style='color:{color};'>Prediction:</b> <span style='color:{color};'>{label}</span></p>
                <p><b style='color:red;'>Spam Probability:</b> <span style='color:red;'>{spam * 100:.2f}%</span></p>
                <p><b style='color:green;'>Ham Probability:</b> <span style='color:green;'>{ham * 100:.2f}%</span></p>
            </div>
        </details>
    """
    st.markdown(expander_html, unsafe_allow_html=True)

    # Create a label for spam found by MultinomialNB and move spam emails to that label
    if prediction == 1:
        # Create the label if it doesn't exist
        label_name = "spam_found_by_mnb"
        # List all labels to check if it already exists
        result, labels = mail.list()
        label_exists = check_label_exists(label_name)
        if not label_exists:
            try:
                mail.create(label_name)
            except mail.error as e:
                st.warning(f"Could not create label '{label_name}': {e}")

        # Copy the email to the new label
        mail.copy(eid, label_name)