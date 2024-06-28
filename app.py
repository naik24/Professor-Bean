import subprocess
subprocess.run('pip install -r requirements.txt', shell = True)

import gradio as gr
import os
from PIL import Image
import numpy as np
from transformers import pipeline
import transformers
transformers.logging.set_verbosity_error()
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_fireworks import ChatFireworks
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console
from rich.markdown import Markdown
from transformers import AutoModelForImageClassification, AutoImageProcessor
from langchain import HuggingFacePipeline

transformers.logging.set_verbosity_error()



def image_to_query(image):
    """
    input: Image

    function: Performs image classification using fine-tuned model

    output: Query for the LLM
    """
    #image = Image.open(image)
    #image = cv2.imread(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image =  Image.fromarray(image)

    model = AutoModelForImageClassification.from_pretrained("nprasad24/bean_classifier", from_tf=True)
    image_processor = AutoImageProcessor.from_pretrained("nprasad24/bean_classifier")

    classifier = pipeline("image-classification", model=model, image_processor=image_processor)

    scores = classifier(image)

    # Get the dictionary with the maximum score
    max_score_dict = max(scores, key=lambda x: x['score'])

    # Extract the label with the maximum score
    label_with_max_score = max_score_dict['label']

    # script to check if the image uploaded is indeed a leaf or not
    counter = 0
    for ele in scores:
        if 0.2 <= ele['score'] <= 0.4:
            counter += 1

    if label_with_max_score == 'healthy' and counter != 3:
        query = "The plant is healthy. Give tips on maintaining the plant"
    elif label_with_max_score == 'bean_rust' and counter != 3:
        query = "The detected disease is bean rust. Explain the disease"
    elif label_with_max_score == 'angular_leaf_spot' and counter != 3:
        query = "The detected disease is angular leaf spot. Explain the disease"
    else:
        query = "Given image is not of a plant."

    return query

def ragChain():
    """
    function: creates a rag chain

    output: rag chain
    """
    loader = TextLoader("knowledgeBase.txt")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)

    vectorstore = vectorstore = FAISS.load_local("faiss_index", embeddings = HuggingFaceEmbeddings(), allow_dangerous_deserialization = True)
    retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k": 5})

    api_key = os.getenv("APIKEY")
    llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct", api_key = api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a knowledgeable agricultural assistant. If a disease is detected, you have to give information on the disease.
                If the image is not of a plant, ask human to upload image of a plant and stop generating any response.
                If the plant is healthy, just give maintenance tips. """
            ),
            (
                "human",
                """Provide information about the leaf disease in question in bullet points.
                Start your answer by mentioning the disease (if any) or healthy in this format: 'Condition: disease name'.
                """,
            ),

            ("human", "{context}, {question}"),
        ]
    )

    rag_chain = (
    {
    "context": retriever,
    "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
    )

    return rag_chain

def generate_response(rag_chain, query):
    """
    input: rag chain, query

    function: generates response using llm and knowledge base

    output: generated response by the llm
    """
    #return Markdown(rag_chain.invoke(f"{query}"))
    return rag_chain.invoke(f"{query}")

def main(image):
    console = Console()

    query = image_to_query(image)
    chain = ragChain()
    output = generate_response(chain, query)
    return output

#main('test2.jpeg')

title = "Professor Bean: The Bean Disease Expert"
description = "Professor Bean is an agricultural expert. He will guide you on how to protect your plants from bean diseases"
app = gr.Interface(fn=main, 
                   inputs="image", 
                   outputs="text", 
                   title=title,
                   description=description,
                   examples=[["sampleImages/sample1.jpg"], ["sampleImages/sample2.jpg"],["sampleImages/sample3.jpg"], ["sampleImages/sample4.jpeg"]]
        )
app.launch(share=True)