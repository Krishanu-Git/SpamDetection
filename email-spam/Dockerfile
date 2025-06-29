FROM python:3.10-slim
WORKDIR /email_spam_detection
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt punkt_tab stopwords
COPY . /email_spam_detection
EXPOSE 8503
CMD ["streamlit", "run", "email-spam/email_spam_classifier.py", "--server.port=8503"]