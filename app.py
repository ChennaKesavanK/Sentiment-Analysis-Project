import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import InferenceClient
import matplotlib.pyplot as plt

# Load sentiment analysis model
sentiment_tokenizer = AutoTokenizer.from_pretrained('karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp')
sentiment_model = AutoModelForSequenceClassification.from_pretrained('karimbkh/BERT_fineTuned_Sentiment_Classification_Yelp')

# Initialize the Hugging Face Inference client with your API key
client = InferenceClient(api_key="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")

def predict_sentiment(review):
    inputs = sentiment_tokenizer(review, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "Positive" if predicted_class == 1 else "Negative"

def generate_contextual_suggestion(reviews):
    # Prepare messages with the reviews classified by the sentiment model
    messages = [
        {"role": "user", "content": f"Here are some reviews classified as positive or negative:\n{reviews}\nSuggest areas for improvement and how the restaurant can take advantage of the positives."}
    ]

    # Call the model via Hugging Face API
    stream = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Replace with the desired model name
        messages=messages,
        max_tokens=500,
        stream=True
    )
    
    # Collect the generated response
    suggestion = ""
    for chunk in stream:
        suggestion += chunk.choices[0].delta.content
    return suggestion.strip()

st.title("Sentiment Analysis App")

# File uploader for CSV or text files
uploaded_file = st.file_uploader("Upload a CSV or TXT file with reviews", type=["csv", "txt"])

if uploaded_file is not None:
    # Initialize lists for sentiments
    sentiments = []
    reviews = []

    # Read the file depending on its format
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        if 'review' in df.columns:
            reviews = df['review'].tolist()
        else:
            st.error("CSV file must contain a 'review' column.")
    elif uploaded_file.name.endswith('.txt'):
        reviews = uploaded_file.read().decode('utf-8').splitlines()
    else:
        st.error("Unsupported file type.")

    # Process reviews and predict sentiments
    for review in reviews:
        prediction = predict_sentiment(review)
        sentiments.append({"review": review, "sentiment": prediction})

    # Create DataFrame with reviews and sentiments
    if sentiments:
        result_df = pd.DataFrame(sentiments)

        # Now add color-coded sentiments in the DataFrame for display
        result_df['Sentiment'] = result_df['sentiment'].apply(lambda x: f"<span style='color: {'green' if x == 'Positive' else 'red'};'>{x}</span>")
        
        # Display reviews and sentiments in a table
        st.markdown("**Reviews and Predictions**")
        st.write(result_df.to_html(escape=False), unsafe_allow_html=True)  # Use HTML to render colored text

        # Call the new model for suggestions
        suggestion_input = "\n".join([f"Review: {row['review']}, Sentiment: {row['sentiment']}" for index, row in result_df.iterrows()])
        contextual_suggestion = generate_contextual_suggestion(suggestion_input)

        st.markdown("**Overall Suggested Improvement**")
        st.write(contextual_suggestion)  # Display the suggestion as a paragraph

        # Pie Chart Visualization
        st.markdown("**Sentiment Distribution**")
        sentiment_counts = pd.Series([row['sentiment'] for row in sentiments]).value_counts()
        plt.figure(figsize=(3, 3))
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title("Sentiment Distribution")
        st.pyplot(plt)
