# SpamDetection
An end-to-end project for email and sms spam detection system

# How to build the docker image for email spam and run the container
1. go to SpamDetection directory 
2. docker build -f email-spam/Dockerfile -t email_spam_check .
3. docker run -p 8585:8503 email_spam_check
4. open http://localhost:8585 to view the results

# How to build the docker image for email spam and run the container
1. go to SpamDetection directory 
2. docker build -f sms-spam/Dockerfile -t sms_spam_check .
3. docker run -p 8586:8504 sms_spam_check
4. open http://localhost:8586 to predict the sms


# Evaluation Metrices Used:
1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. Confusion Matrix

# How to run email spam classifier
1. python3 email_spam_classifier