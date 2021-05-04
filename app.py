import pickle
from tensorflow.keras.models import load_model
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

def predict(message):
 model=load_model('/content/drive/MyDrive/Projects/Twitter_sentiment_analysis/sentient_model1.h5')
 with open("/content/drive/MyDrive/Projects/Twitter_sentiment_analysis/tokenizer.pickle", "rb") as handle:
  tokenizer = pickle.load(handle)
 x_1 = tokenizer.texts_to_sequences([message])
 x_1 = pad_sequences(x_1, maxlen=500)
 predictions = model.predict(x_1)[0][0]
 ori = 1 if predictions >0.6 else 0
 return (predictions,ori)
st.title("Movie Review Sentiment Analyzer")
message = st.text_area("Enter Review",'Type Here ..')
if st.button('Analyze'):
  with st.spinner('Analyzing the text …'):
    prediction=predict(message)
    if prediction > 0.6:
      st.success('Positive review with {:.2f} confidence'.format(prediction))
      st.balloons()
    elif prediction <0.4:
      st.error('Negative review with {:.2f} confidence'.format(1-prediction))
    else:
      st.warning('Not sure! Try to add some more words')