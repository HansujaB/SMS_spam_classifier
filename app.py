import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.tokenize import word_tokenize , sent_tokenize
ps=PorterStemmer()

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


