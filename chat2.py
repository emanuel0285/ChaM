import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV file
data = pd.read_csv("AI.csv")

# Define a function to clean the text data
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Remove numbers
    text = re.sub('\d+', '', text)
    # Remove extra spaces
    text = re.sub(' +', ' ', text)
    return text

# Apply the cleaning function to the 'question' column
data['Question'] = data['Question'].apply(clean_text)

# Apply the cleaning function to the 'answer' column
data['Answer'] = data['Answer'].apply(clean_text)

# Initialize a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Create a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(data['Question'])

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Define a function to get the most similar answer for a given question
def get_answer(question):
    # Clean the question text
    question = clean_text(question)
    # Transform the question into a TF-IDF vector
    question_vec = vectorizer.transform([question])
    # Calculate the cosine similarity between the question and all the questions in the dataset
    sim_scores = cosine_similarity(question_vec, tfidf_matrix)
    # Get the index of the most similar question
    idx = sim_scores.argmax()
    # Return the corresponding answer
    return data.iloc[idx]['Answer']

# Define the conversation variable
conversation = []

# Define a function to generate and present the report
def generate_report():
    # Perform necessary calculations or queries to generate the report
    # This is just a placeholder example
    report_data = {
        "Report Name": "AI Chatbot Report",
        "Total Questions": len(data),
        "Unique Answers": data["Answer"].nunique(),
        "Usage Report": {
            # Add usage report data
            "Total Conversations": len(conversation),
            "Most Recent Conversations": conversation[-5:][::-1]
        }
        "Performance Report": {
            # Add performance report data
        },
        "Error Report": {
            # Add error report data
        },
        "User Feedback Report": {
            # Add user feedback report data
        },
        "Trend Analysis Report": {
            # Add trend analysis report data
        },
        "Knowledge Gap Report": {
            # Add knowledge gap report data
        },
        "Performance Comparison Report": {
            # Add performance comparison report data
        },
        "Conversational Flow Report": {
            # Add conversational flow report data
        }
    }

    # Present the report
    st.write("Generated Report:")
    st.write("Report Name:", report_data["Report Name"])
    st.write("Total Questions:", report_data["Total Questions"])
    st.write("Unique Answers:", report_data["Unique Answers"])
    st.write("Usage Report:", report_data["Usage Report"])
    st.write("Performance Report:", report_data["Performance Report"])
    st.write("Error Report:", report_data["Error Report"])
    st.write("User Feedback Report:", report_data["User Feedback Report"])
    st.write("Trend Analysis Report:", report_data["Trend Analysis Report"])
    st.write("Knowledge Gap Report:", report_data["Knowledge Gap Report"])
    st.write("Performance Comparison Report:", report_data["Performance Comparison Report"])
    st.write("Conversational Flow Report:", report_data["Conversational Flow Report"])


# Define a function to run the chatbot
def run_chatbot():
    st.write("Hi, I'm a chatbot. How can I help you?")

    form_key = 'question-form'
    with st.form(key=form_key):
        question = st.text_input(">", key=f"{form_key}-input")
        form_submit = st.form_submit_button("Submit")

        if form_submit:
            if question.lower() in ['help', 'h']:
                display_help()
                conversation.append((question, "Help"))
            elif question.lower() in ['exit', 'e']:
                st.write("Goodbye!")
                conversation.append((question, "Goodbye"))
            elif question.lower() in ['report', 'r']:
                generate_report()
                conversation.append((question, "Report Generated"))
            else:
                answer = get_answer(question)
                if answer:
                    st.write(answer)
                    conversation.append((question, answer))
                else:
                    default_answer = "Sorry, I don't understand. Do you need help? Type 'help' or 'h' for more information. Type 'exit' or 'e' to quit."
                    st.write(default_answer)
                    conversation.append((question, "Unknown"))

# Define a function to display the help message
def display_help():
    st.write("I'm a chatbot that can answer your questions. Here are some things you can ask me:")
    st.write("1. Ask me a question, and I'll try my best to answer it.")
    st.write("2. Type 'help' or 'h' to see this message again.")
    st.write("3. Type 'exit' or 'e' to quit.")
    st.write("4. Type 'report' or 'r' to see the report.")

run_chatbot()
