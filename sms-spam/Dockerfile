FROM python:3.10-slim
WORKDIR /SpamDetection
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt punkt_tab stopwords
COPY . /SpamDetection
# Expose the port that Streamlit uses
EXPOSE 8504
CMD ["streamlit","run", "sms-spam/sms_spam_detection.py", "--server.port=8504"]