import streamlit as st
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
from nltk.stem.porter import PorterStemmer
import google.generativeai as genai
import os

# Setup Gemini API
genai.configure(api_key=os.getenv("api_key"))

# Load Q&A data
df = pd.read_csv("ques_ans.csv")
dataset = dict(zip(df["QUESTIONS"], df["ANSWERS"]))
questions = list(dataset.keys())
answers = list(dataset.values())

# NLP prep
stemmer = PorterStemmer()

def stem(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

stemmed_questions = [stem(q) for q in questions]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(stemmed_questions)

# Utilities
def clean_query(text):
    return text.strip().rstrip(string.punctuation)

def correct_spelling(query):
    words = query.split()
    corrected = []
    for word in words:
        match = process.extractOne(word, " ".join(questions).split(), scorer=fuzz.ratio)
        corrected.append(match[0] if match and match[1] >= 80 else word)
    return " ".join(corrected)

def postoffice_related(text):
    keywords = [
        "post office", "mail", "parcel", "stamp", "delivery", "shipping", "courier", "postal", 
        "package", "tracking", "address", "pincode", "speed post", "money order", "letter", 
        "box", "mailbox", "return", "sender", "postage", "envelope", "pickup", "drop-off", 
        "priority", "customs", "international", "domestic"
    ]
    return any(k in text.lower() for k in keywords)

def fetch_gemini_answer(question):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Answer the following question: {question}")
    return response.text

def get_answer(user_query, threshold=0.75):
    cleaned = clean_query(user_query)
    corrected = correct_spelling(cleaned)
    stemmed = stem(corrected)
    user_vec = vectorizer.transform([stemmed])
    sims = cosine_similarity(user_vec, question_vectors).flatten()
    best_idx = np.argmax(sims)
    max_sim = sims[best_idx]

    if max_sim >= threshold:
        return answers[best_idx]
    elif postoffice_related(cleaned):
        gemini_response = fetch_gemini_answer(cleaned)
        # Auto-update local dataset
        dataset[cleaned] = gemini_response
        pd.DataFrame(dataset.items(), columns=["QUESTIONS", "ANSWERS"]).to_csv("ques_ans.csv", index=False)
        return gemini_response
    else:
        return "Sorry, I am not designed to answer this question."

# Streamlit UI
st.set_page_config(page_title="Query Bot 💬", page_icon="📮")
st.title("📮 Post Office Q&A Assistant")
st.write("Ask me a question about post office services!")

user_input = st.text_input("Your question:", placeholder="e.g., How long does speed post take?")

if user_input:
    with st.spinner("Thinking..."):
        response = get_answer(user_input)
    st.markdown("### 🧠 Answer")
    st.success(response)
    st.caption("Tip: Ask about tracking, parcels, delivery, or mail services.")
