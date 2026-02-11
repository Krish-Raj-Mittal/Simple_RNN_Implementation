import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the Pre-trained model With RelU activation
model = load_model('simple_rnn_model.h5')

#Step 2: Helper function to decode reviews
#Function to Decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  
    padded_review = tf.keras.preprocessing.sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#Step 3 : Prediction function

def predict_sentiment(text):
    processed_input = preprocess_text(text)
    prediction = model.predict(processed_input)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
    return sentiment, prediction[0][0]


###### Designing Streamlit Interface ######
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to Classify it as (Positive/Negative).")

# User input
user_input = st.text_area("Enter your movie review here:")

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    # Make my Prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'

    # Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")
    
else:
    st.write("Please enter a review and click 'Classify' to see the result.")