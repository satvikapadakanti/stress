# Stress Detection using BERT + Gradio

A simple web app that detects emotional tone from user input using a pretrained BERT model (`cardiffnlp/twitter-roberta-base-emotion`). Deployed with Gradio.

## Features
- Predicts emotions like joy, anger, sadness, optimism
- Displays if text indicates STRESS or NO STRESS
- Built with Gradio and Hugging Face Transformers

## Run Locally

```bash
pip install -r requirements.txt
python gradio_bert_app.py
