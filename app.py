import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

# Set a consistent NLTK data path
NLTK_PATH = "/opt/render/nltk_data"
os.makedirs(NLTK_PATH, exist_ok=True)
nltk.data.path.append(NLTK_PATH)

# Download necessary NLTK resources
try:
    nltk.download("punkt", download_dir=NLTK_PATH, quiet=True)
    nltk.download("punkt_tab", download_dir=NLTK_PATH, quiet=True)  # Add this specific resource
    nltk.download("stopwords", download_dir=NLTK_PATH, quiet=True)
    print(f"NLTK resources downloaded to {NLTK_PATH}")
except Exception as e:
    st.error(f"Failed to download NLTK resources: {str(e)}")
    print(f"Error downloading NLTK resources: {str(e)}")
    
# Initialize Porter Stemmer
ps = PorterStemmer()

tfidf=pickle.load(open('vectorizer.pkl', 'rb'))
model=pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

def text_transform(text):
  text=text.lower()
  text=nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text=y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

input_sms=st.text_area("Enter the message")
if st.button('Predict'):
  if input_sms.strip() == "":
    st.warning("Please enter a message before predicting.")
  else:
    # Preprocess
    transformed_sms = text_transform(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display Result
    st.header("SPAM" if result == 1 else "NOT SPAM")


