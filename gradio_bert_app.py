import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = ['anger', 'joy', 'optimism', 'sadness']

def detect_stress(text):
    encoded = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded)
    scores = softmax(output.logits.numpy()[0])
    label_scores = dict(zip(labels, scores))
    top_emotion = max(label_scores, key=label_scores.get)
    stress_status = "STRESS" if top_emotion in ['anger', 'sadness'] else "NO STRESS"
    confidence = round(label_scores[top_emotion], 2)
    return f"Emotion: {top_emotion.upper()} ({confidence}) → {stress_status}"

# Gradio UI
demo = gr.Interface(
    fn=detect_stress,
    inputs=gr.Textbox(lines=4, placeholder="Enter your feelings here..."),
    outputs="text",
    title="Stress Detection Using Emotion AI 🤖",
    description="Type something, and the app will detect if you're stressed based on emotional cues."
)

demo.launch()
