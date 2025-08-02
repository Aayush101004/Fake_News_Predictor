import os  # Keep for os.getenv if using .env, otherwise can remove
import re

# --- NLTK Data Download (Most Robust for Streamlit Cloud) ---
# This ensures stopwords are downloaded before any cached functions try to use them.
import nltk
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import wikipedia  # For Wikipedia search
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import \
    confusion_matrix  # For terminal output of confusion matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Dense, Dropout, Embedding,
                                     GlobalAveragePooling1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# -----------------------------------------------------------

@st.cache_data
def load_and_preprocess_data(true_file_path='True.csv', false_file_path='Fake.csv', delimiter=','):
    try:
        true_df = pd.read_csv(true_file_path, sep=delimiter)
        fake_df = pd.read_csv(false_file_path, sep=delimiter)
    except FileNotFoundError as e:
        st.error(f"Error: One or more data files not found ({e}). Please ensure 'True.csv' and 'Fake.csv' are in the same directory as 'app.py'.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data files: {e}. Please check the delimiter and file formats.")
        st.stop()

    true_df['label'] = 0
    fake_df['label'] = 1

    if 'title' in true_df.columns and 'title' in fake_df.columns:
        text_col = 'title'
    elif 'text' in true_df.columns and 'text' in fake_df.columns:
        text_col = 'text'
    else:
        st.error("No common 'title' or 'text' column found in both True.csv and Fake.csv.")
        st.stop()

    news_df = pd.concat([true_df, fake_df], ignore_index=True)
    news_df.dropna(subset=[text_col], inplace=True)

    ps = PorterStemmer()
    # Resolve stopwords list ONCE within the cached function
    english_stopwords_list = stopwords.words('english')

    def stemming(content):
        if not isinstance(content, str):
            return ""
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        # Use the resolved stopwords list
        stemmed_content = [ps.stem(word) for word in stemmed_content if word not in english_stopwords_list]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content

    news_df[text_col] = news_df[text_col].astype(str).apply(stemming)
    news_df = news_df.rename(columns={text_col: 'title'})
    return news_df

@st.cache_resource
def train_model(news_df_processed):
    X = news_df_processed['title'].values
    y = news_df_processed['label'].values

    vocab_size = 10000
    embedding_dim = 100
    max_len = 100
    trunc_type = 'post'
    padding_type = 'post'

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)

    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, maxlen=max_len, padding=padding_type, truncating=trunc_type)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, stratify=y, random_state=2)

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )
    return model, tokenizer, max_len, padding_type, trunc_type, X_test, y_test

news_df_processed = load_and_preprocess_data()

model, tokenizer, max_len, padding_type, trunc_type, X_test_cached, y_test_cached = train_model(news_df_processed)

y_pred_proba = model.predict(X_test_cached)
y_pred = (y_pred_proba > 0.5).astype(int)

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown("""
    <h1 style='
        text-align: center;
        font-weight: bold;
        font-size: 3em;
    '>
        📰 <span style='
            background: linear-gradient(to right, #4CAF50, #81C784);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
        '>Fake News Detector</span>
    </h1>
""", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em;'>Enter a news article title below to determine if it's likely real or fake using a deep learning model.</p>", unsafe_allow_html=True)

input_text = st.text_area(
    'Enter News Article Title Here:',
    height=150,
    help="Paste the full title of the news article you want to check for authenticity."
)

def predict_news_category(text):
    if not text.strip():
        return None, None

    ps_local = PorterStemmer()
    # Resolve stopwords list ONCE within the prediction function call
    english_stopwords_list_pred = stopwords.words('english')

    def local_stemming(content):
        if not isinstance(content, str):
            return ""
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        # Use the resolved stopwords list
        stemmed_content = [ps_local.stem(word) for word in stemmed_content if word not in english_stopwords_list_pred]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content

    processed_text = local_stemming(text)

    input_sequence = tokenizer.texts_to_sequences([processed_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_len, padding=padding_type, truncating=trunc_type)

    prediction_proba = model.predict(input_padded)[0][0]
    prediction_label = 1 if prediction_proba >= 0.5 else 0

    return prediction_label, prediction_proba

def perform_wikipedia_search(query):
    try:
        summary = wikipedia.summary(query, sentences=3, auto_suggest=True, redirect=True)
        page = wikipedia.page(query, auto_suggest=True, redirect=True)
        return [{
            'title': page.title,
            'url': page.url,
            'snippet': summary
        }]
    except wikipedia.DisambiguationError as e:
        st.warning(f"Your query is ambiguous. Please be more specific. Options: {', '.join(e.options[:5])}")
        return []
    except wikipedia.PageError:
        st.warning("No Wikipedia page found for this query.")
        return []
    except Exception as e:
        st.error(f"Wikipedia error: {e}")
        return []

def generate_google_search_url(query):
    base_url = "https://www.google.com/search?q="
    encoded_query = query.replace(" ", "+")
    return f"{base_url}{encoded_query}"

if st.button('Analyze News', help="Click to analyze the entered news title."):
    if input_text.strip():
        label, proba = predict_news_category(input_text)

        st.markdown("---")
        if label is not None:
            st.subheader("Analysis Result:")
            st.write(f"**Original Title:** \"{input_text}\"")

            if label == 1: # Predicted as FAKE
                st.markdown("<h3 style='color: #FF6347;'>🚨 This News is Likely FAKE!</h3>", unsafe_allow_html=True)
                st.write(f"Confidence (Fake): **{proba*100:.2f}%**")
                st.write(f"Confidence (Real): {(1-proba)*100:.2f}%")

                st.subheader("For Fact-Checking:")
                # For Fake News, only provide Google search for fact-checking
                fact_check_query = f"fact check {input_text}"
                fact_check_url = generate_google_search_url(fact_check_query)
                st.markdown(f"Click [here]({fact_check_url}) to search Google for fact-checks on this news.")
                st.info("This link will open a Google search in a new tab.")

            else: # Predicted as REAL
                st.markdown("<h3 style='color: #28A745;'>✅ This News is Likely REAL!</h3>", unsafe_allow_html=True)
                st.write(f"Confidence (Real): **{(1-proba)*100:.2f}%**")
                st.write(f"Confidence (Fake): {proba*100:.2f}%")

                st.subheader("Related Information:")
                # For Real News, try Wikipedia first
                wiki_results = perform_wikipedia_search(input_text)
                if wiki_results:
                    for i, result in enumerate(wiki_results):
                        st.write(f"**[{result['title']}]({result['url']})**")
                        st.write(result['snippet'])
                        st.markdown("---")
                else:
                    st.write("No direct Wikipedia summary found.")

                # Always provide the Google search link for Real News
                similar_news_url = generate_google_search_url(input_text)
                st.markdown(f"Click [here]({similar_news_url}) to search Google for more information.")
                st.info("This link will open a Google search in a new tab.")

            st.markdown("---")
            st.info("Disclaimer: This model is for demonstrative purposes and may not be 100% accurate. Always verify information from multiple reliable sources.")
        else:
            st.warning("Please enter some text to analyze.")
    else:
        st.warning("Please enter some text to analyze.")
        st.warning("Please enter some text to analyze.")
