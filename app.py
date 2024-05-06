import streamlit as st
import spacy
from textblob import TextBlob
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Download NLTK data
nltk.download('punkt')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def correct_text(text):
    """Processes text for grammar, spelling correction, punctuation, and sentiment analysis."""
    
    # Correct spelling using TextBlob
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    
    # Process text using spaCy
    doc = nlp(corrected_text)
    
    # Get corrected and analyzed text
    corrected_text = " ".join([token.text for token in doc])
    
    # Sentiment analysis using VADER
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(corrected_text)
    
    return corrected_text, sentiment_scores

def summarize_text(text, num_sentences=3):
    """Summarize the text to provide a concise overview of the main points."""
    
    sentences = sent_tokenize(text)
    word_frequencies = Counter()
    
    for word in nltk.word_tokenize(text):
        if word.lower() not in STOP_WORDS:
            word_frequencies[word.lower()] += 1
    
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/max_frequency)
    
    sentence_scores = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequencies.keys():
                if len(sentence.split(' ')) < 30:
                    if sentence not in sentence_scores.keys():
                        sentence_scores[sentence] = word_frequencies[word]
                    else:
                        sentence_scores[sentence] += word_frequencies[word]
    
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    
    return summary

# Create the Streamlit app structure
st.title("Advanced Grammar, Spelling, and Sentiment Checker")
st.subheader("Enter your text for correction:")

# Get user input
text_input = st.text_area("", height=200)

# Process text on button click
if st.button("Check"):
    corrected_text, sentiment_scores = correct_text(text_input)
    
    st.success("Corrected Text:")
    st.write(corrected_text)
    
    # Show Sentiment Analysis Result
    st.subheader("Sentiment Analysis:")
    st.write(f"Positive: {sentiment_scores['pos']}")
    st.write(f"Negative: {sentiment_scores['neg']}")
    st.write(f"Neutral: {sentiment_scores['neu']}")
    
    # Show Summarized Text
    st.subheader("Summary:")
    summary_text = summarize_text(corrected_text)
    st.write(summary_text)


# Show Word Frequency Analysis
st.subheader("Word Frequency Analysis:")
word_frequencies = Counter(nltk.word_tokenize(corrected_text.lower()))

# Filter out stopwords and punctuation
filtered_word_frequencies = {word: freq for word, freq in word_frequencies.items() if word.isalpha() and word.lower() not in STOP_WORDS}

# Display the top 5 most common meaningful words
common_words = Counter(filtered_word_frequencies).most_common(5)
for word, frequency in common_words:
    st.write(f"{word}: {frequency}")




#Thiis sentennce hass speling and gramar mistakes. The orijinal texxt is nott correctt.
