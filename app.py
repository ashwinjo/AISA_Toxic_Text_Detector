import requests
import gradio as gr
import json
import plotly.graph_objects as go
from webserver import start_web_server

# Define the API endpoints
get_random_tweet_url = "http://127.0.0.1:5050/random_tweet"
openapi_url = "http://127.0.0.1:5050/test_tweet_openapi"
huggingface_url = "http://127.0.0.1:5050/test_tweet_huggingface"

def _get_random_tweet():
    response = requests.get(get_random_tweet_url)
    return response.json()['random_tweet']

def _test_tweet_openapi(tweet):
    response = requests.post(openapi_url, json={"tweet": tweet})
    return response.json()
    # response['categories'].json(), response['category_score'].json()

def _test_tweet_huggingface(tweet):
    response = requests.post(huggingface_url, json={"tweet": tweet})
    return response.json()

def _shutdown_server():
    return requests.get("http://127.0.0.1:5050/shutdown")


mksdown = """# 😃 Welcome To The Friendly Text Moderation for Twitter (X) Posts  
### 🔍 Identify 13 Categories of Text Toxicity  

> 🚀 This **NLP-powered AI** aims to detect and prevent **profanity, vulgarity, hate speech, violence, sexism, and offensive language** in tweets.  
> 🛡️ **Not an act of censorship** – the UI allows readers (excluding young audiences) to click on a label to reveal toxic messages.  
> 🎯 **Goal**: Foster a safer, more respectful online space for **you, your colleagues, and your family**.  

---

## 🛠️ How to Use This App?  
1️⃣ **Enter your tweet** (or use "Populate Random Tweet" to load a harmful tweet from a Kaggle dataset).  
2️⃣ **Click "Measure Toxicity OpenAPI"** to analyze toxicity across 13 categories, visualized as a **horizontal bar graph**.  
3️⃣ **Click "Measure Toxicity HF"** to get a **JSON-based** safe/unsafe result with toxicity percentages using **Hugging Face**.  

---

## 📌 AI Models Used  
- 🧠 **OpenAI’s 'omni-moderation-latest' model** for multi-category toxicity detection.  
- 🤖 **Hugging Face’s 'unitary/toxic-bert' model** for binary (Safe/Unsafe) classification.  
- 🔬 **Understands context, nuance, and intent** – beyond just swear words!  

---

## 📊 Toxicity Categories (13)  
1️⃣ **Sexual**  
2️⃣ **Harassment**  
3️⃣ **Violence**  
4️⃣ **Hate**  
5️⃣ **Illicit**  
6️⃣ **Harassment/Threatening**  
7️⃣ **Self-Harm**  
8️⃣ **Sexual/Minors**  
9️⃣ **Self-Harm/Intent**  
🔟 **Self-Harm/Instructions**  
1️⃣1️⃣ **Illicit/Violent**  
1️⃣2️⃣ **Hate/Threatening**  
1️⃣3️⃣ **Violence/Graphic**  

---

## 📝 Example Hugging Face Output  
```json
{
    "toxicity": "unsafe",
    "%Toxic": 65.95,
    "%Safe": 34.05
}

* Open API analyzes tweet for 13 categories and displays them with %
* The real-world dataset is from the "Toxic Tweets Dataset" (https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset/data)
---
# 🌟 "AI Solution Architect" Course by ELVTR
"""

# Function to get toxicity scores from OpenAI
def get_toxicity_openai(tweet):
    open_api_answer = _test_tweet_openapi(tweet)
    open_api_answer['category_scores']['IS THIS TWEET TOXIC'] = open_api_answer['flagged']
    # Convert scores to percentages
    categories = list(open_api_answer['category_scores'].keys())
    scores = [score * 100 for score in open_api_answer['category_scores'].values()]  # Multiply by 100 to get percentage

    # Create the bar graph using Plotly
    fig = go.Figure(go.Bar(
        x=categories,
        y=scores,
        text=[f"{score:.2f}%" for score in scores],  # Format the text as percentage
        textposition='auto',  # Position the text inside the bars
        marker_color=['red' if score > 90 else 'green' for score in scores],  # Color red if > 50%
    ))

    # Update layout for better appearance
    fig.update_layout(
        title="Toxicity Categories",
        xaxis_title="Category",
        yaxis_title="Percentage (%)",
        showlegend=False
    )

    # Return the figure object to be displayed in Gradio
    return fig


# Function to get toxicity scores from Hugging Face
def get_toxicity_hf(tweet):
    hugging_face_answer = _test_tweet_huggingface(tweet)
    return hugging_face_answer[0]


def get_toxicity_hf(tweet):
    hugging_face_answer = _test_tweet_huggingface(tweet)
    print(hugging_face_answer)
    score = hugging_face_answer[0]['score']*100
    if score <= 60:
       return json.dumps({"toxicity": "safe", "%Toxic": score, "%safe": (100-score)}, indent=4)
    else:
       return json.dumps({"toxicity": "unsafe", "%Toxic": score, "%safe": (100-score)}, indent=4)

# Random Tweet Generator
def get_random_tweet():
    return _get_random_tweet()

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown(mksdown)
    user_input = gr.Textbox(label="Paste your paragraph (2-10 lines)", lines=5)

    with gr.Row():
        analyze_btn = gr.Button("Measure Toxicity OpenAPI")
        analyze_btn_hf = gr.Button("Measure Toxicity HF")
        random_tweet_btn = gr.Button("Populate Random Tweet")

    toxicity_output_json = gr.Code(label="Formatted Toxicity JSON", language="json")
    toxicity_output = gr.Plot()
    
    analyze_btn_hf.click(get_toxicity_hf, inputs=[user_input], outputs=[toxicity_output_json])
    analyze_btn.click(get_toxicity_openai, inputs=[user_input], outputs=[toxicity_output])
    random_tweet_btn.click(get_random_tweet, outputs=user_input)

if __name__ == "__main__":
    start_web_server()
    demo.launch(debug=True)

