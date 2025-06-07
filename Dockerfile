FROM ubuntu:24.04
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean
RUN pip install --upgrade pip
# Set Python3 as the default python
RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /SpamDetection
COPY . /SpamDetection
RUN pip install --no-cache-dir -r requirements.txt
# Expose the port that Streamlit uses
EXPOSE 8501
CMD ["stream","run", "spam_detection.py"]