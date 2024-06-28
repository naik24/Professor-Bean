# Professor Bean 👨‍🏫: Leaf Disease Diagnosis with Image Classification with Image Classification and LLM-Based Retrieval Augmented Generation [![access-demo-here](https://img.shields.io/badge/Access%20Demo-Here-1f425f.svg)](https://huggingface.co/spaces/nprasad24/Professor-Bean)

## Introduction

In modern agriculture, early detection and precise disease diagnosis are crucial for maintaining crop health and ensuring high yields. Rapid identification of plant diseases can prevent the spread of infections, saving significant time and resources. Moreover, accurate diagnosis allows for targeted treatment, minimizing the use of harmful chemicals and promoting sustainable farming practices.

Professor Bean is a comprehensive AI-powered system designed to diagnose leaf diseases and provide expert recommendations. By combining advanced image classification with language models, Professor Bean offers a robust solution for farmers and agricultural experts.

## Image Classification Model

At the core of Professor Bean is an image classification model that identifies leaf diseases. We employed the "google/vit-base-patch16-224-in21k" model, a variant of the Vision Transformer (ViT) architecture. This model achieved an impressive accuracy of 99%, ensuring reliable disease classification into three categories: "Angular Leaf Spot," "Bean Rust," and "Healthy." The model is hosted on huggingface <a href = "https://huggingface.co/nprasad24/bean_classifier">here</a>

## Language Model and Retrieval Augmented Generation

After image classification, Professor Bean uses the mixtral-8x7b-instruct model by MistralAI to generate detailed responses about the detected disease. To enhance the accuracy and relevance of these responses, we implemented a Retrieval Augmented Generation (RAG) framework using Langchain. This framework allows the language model to access a curated knowledge base, which we meticulously prepared, containing information about various leaf diseases, their causes, and potential remedies.

## User Interface
To make the system user-friendly, we developed the interface using Gradio. This tool allows users to upload leaf images, receive real-time disease diagnoses, and access comprehensive information about the disease and suggested treatments.

## System Workflow

1. Image Upload: Users upload a leaf image through the Gradio interface.
2. Image Classification: The "google/vit-base-patch16-224-in21k" model classifies the image into "Angular Leaf Spot," "Bean Rust," or "Healthy."
3. Query Generation: Based on the classification result, a query is generated for the mixtral-8x7b-instruct model.
4. Retrieval Augmented Generation: Using Langchain, the language model accesses the knowledge base for relevant information.
5. Response Generation: The language model generates a detailed response with causes and remedies for the disease.
6. User Feedback: The response is displayed on the Gradio interface, providing users with actionable insights and expert advice.

## Getting Started
To get started with Professor Bean, follow these steps:

1. Clone the repository
```python
git clone https://github.com/yourusername/professor-bean.git
```

2. Install Dependencies:
```python
cd professor-bean
pip install -r requirements.txt
```

3. Run the application
```
python app.py
```
