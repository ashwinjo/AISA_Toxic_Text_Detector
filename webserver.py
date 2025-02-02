import pandas as pd
import random
import nest_asyncio
from fastapi import FastAPI
from uvicorn import run , Config, Server
import openai
import threading
from pydantic import BaseModel
import os

# Allow FastAPI to run in the notebook event loop
nest_asyncio.apply()
# Initialize FastAPI app
app = FastAPI()

# Load the CSV file into a DataFrame
df = pd.read_csv('cleaned_toxic_tweets.csv')  # Replace with the actual path to your CSV file

# Ensure the 'tweet' column exists in the DataFrame
if 'cleaned_tweet' not in df.columns:
    raise ValueError("The CSV file must contain a 'tweet' column")

# Define request body model
class TweetRequest(BaseModel):
    tweet: str

# Endpoint to get a random tweet
@app.get("/random_tweet")
def get_random_tweet():
    # Get a random value from the 'tweet' column
    random_tweet = random.choice(df['cleaned_tweet'].tolist())
    return {"random_tweet": random_tweet}

@app.post("/test_tweet_openapi")
def test_tweet_openapi(request: TweetRequest):
    # Open AI Access
    toxic_comment = request.tweet
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Input: {toxic_comment}")
    ai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = ai_client.moderations.create(input=toxic_comment, model='omni-moderation-latest')
    # Extract the first result
    print("Open API omni-moderation-latest Response:")
    return response.results[0]


@app.post("/test_tweet_huggingface")
def test_tweet_huggingface(request: TweetRequest):
    from transformers import pipeline
    from transformers import pipeline

    # This model only tells toxic / not toxic when using pipeline way of getting results.
    # When using the Hugging Face pipeline for text classification
    # with the unitary/toxic-bert model, it typically provides a simple
    # classification output, such as "toxic" or "non-toxic."
    # This is because the pipeline is designed to give a straightforward result based on the model's predictions.
    toxic_comment = request.tweet
    pipe = pipeline("text-classification", model="unitary/toxic-bert")
    helper = pipe(toxic_comment)
    print("HF model unitary/toxic-bert Response:")
    return helper

# API to shut down the server gracefully
@app.get("/shutdown")
def shutdown_server():
    global server
    if server is not None:
        server.should_exit = True  # Signal Uvicorn to shut down
        return "Server shutting down..."
    return "Server not running"


# Global variable to store Uvicorn server instance
server = None

# Function to run FastAPI in a separate thread
def run_fastapi():
    global server
    config = Config(app, host="127.0.0.1", port=5050, log_level="info")
    server = Server(config)
    server.run()

def start_web_server():
    # Run the FastAPI server in the background thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    print("✅ FastAPI server is running in the background. You can proceed with other cells.")

