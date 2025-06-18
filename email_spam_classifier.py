import pickle
import string
import imaplib
import email
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from functools import lru_cache

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
EMAIL = 'g24ai2013@iitj.ac.in'
PASSWORD = 'jhun lfjc viwt bwej'

# Connect to Gmail IMAP server
mail = imaplib.IMAP4_SSL('imap.gmail.com')
mail.login(EMAIL, PASSWORD)
mail.select('inbox')

# Search for latest 100 emails in inbox
result, data = mail.search(None, 'ALL')
email_ids = data[0].split()
last_50_emails = email_ids[-50:][::-1]

for eid in last_50_emails:
    result, msg_data = mail.fetch(eid, '(RFC822)')
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    subject = msg['subject']
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
                break
    else:
        body = msg.get_payload(decode=True).decode()

    # Combine subject and body for prediction
    email_text = f"{subject} {body}"
    print(f"Email Subject: {subject}")
    print(f"Email Body: {body}")

    # Preprocess and vectorize
    transformed = transform_text(email_text)
    vectorized = tfidf.transform([transformed]).toarray()

    # Predict
    prediction = model.predict(vectorized)[0]
    ham, spam = model.predict_proba(vectorized)[0]
    label = 'Spam' if prediction == 1 else 'Ham'
    print(f"Prediction: {label}")
    print(f"Spam Probability: {spam * 100:.2f}%, Ham Probability: {ham * 100:.2f}%")

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
                print(f"Label '{label_name}' created.")
            except mail.error as e:
                print(f"Could not create label '{label_name}': {e}")
        else:
            print(f"Label '{label_name}' already exists.")

        # Copy the email to the new label
        mail.copy(eid, label_name)
        print(f"Email ID {eid.decode()} moved to label '{label_name}'")
    else:
        print("Email is not spam, no action taken.")

    print("\n\n")


"""
An example of a spam email that might be detected:

Subject: Urgent: Your account has been suspended!

Dear user,

We noticed suspicious activity in your account. Please verify your details immediately to restore access.

Click here to verify: http://fake-verify-site.com

Sincerely,
Security Team
"""