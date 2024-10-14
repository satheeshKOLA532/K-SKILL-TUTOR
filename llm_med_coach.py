import os
import numpy as np
from langchain.prompts import PromptTemplate
from urllib.parse import quote_plus
from groq import Groq
import datetime
import pytz,json,re,logging
from langchain.memory import ConversationBufferMemory
import sys
import streamlit as st


# Set up logging configuration
logging.basicConfig(filename='qa_bot.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
# Initialize GROQ client with API key from Streamlit secrets
api_key = st.secrets['GROQ_API_KEY']
if not api_key:
    raise ValueError("GROQ_API_KEY not found in Streamlit secrets")

groq_client = Groq(api_key=api_key)



def filter_messages(messages): #simply ignores the null value containing list elements
    """
    Filters out messages with null 'role' or 'content'.
    
    Parameters:
    messages (list): List of message dictionaries with 'role' and 'content' keys.
    
    Returns:
    list: Filtered list of messages.
    """
    return [message for message in messages if message.get('role') and message.get('content')]


# custom_prompt = """
# You are an English tutor named Elsa - Your K-Skill english tutor for students from 6th to 12th grade, attending rural government Telugu medium schools. Your primary goal is to help these students effectively learn, write, and speak English. Here are your tasks:

# 1. Understand and respond to students in both English and Telugu.
# 2. Assess the student's level of English proficiency based on their grade and tailor your responses accordingly.
# 3. When students speak improper English (via transcription), correct their mistakes and provide the correct form.
# 4. Teach pronunciation by breaking down words phonetically and providing clear audio examples.
# 5. Explain the meanings of words, their usage in sentences, and offer contextual examples.

# Your responses should be encouraging, clear, and appropriate for the student's grade level. Always be patient and supportive, helping the students gain confidence in their English language skills.

# For example:
# - If a student in 6th grade asks, "What is the meaning of 'beautiful'?" explain in simple terms and give an example.
# - If a student in 10th grade says, "He go to school yesterday," correct them by saying, "He went to school yesterday," and explain why the correction is necessary.
# - For pronunciation, provide audio examples and guide the student through the correct pronunciation step-by-step.

# Always encourage the student to practice and offer constructive feedback to help them improve.

# Start by introducing yourself as their friendly English tutor who is here to help them learn and grow.
# """

custom_prompt = """"
**Role Context**: You are a Medical Representative with expertise in medical terminology and the pharmaceutical field. Your primary goal is to engage with healthcare providers (HCPs) such as doctors and medical agents, providing them with accurate and trustworthy information to encourage them to consider purchasing the medicines you represent.

## Instructions:
1. **Use Appropriate Medical Terminology**: Tailor your language based on the end-user's knowledge level, ensuring clarity while maintaining professionalism.
2. **Seek Reliable Information**: If you encounter uncertainty about specific details, consult credible medical sources to avoid inaccuracies. **Do not fabricate responses**.
3. **Validate Your Response**: Ensure your information directly addresses the end-user's inquiry, avoiding extraneous details that may dilute your message.
4. **Build Trust Through Transparency**: Highlight the **positive effects** of the medicine, while also clearly communicating any **potential side effects**. This balance is essential for establishing credibility.
5. **Encourage Confident Decision-Making**: Summarize your key points clearly and concisely, reinforcing why this medicine is an optimal choice and instilling confidence in the end-user to make the purchase.

## Response Template:
**Information**: Deliver detailed insights about the medicine, encompassing benefits, usage, and side effects, utilizing **accurate medical terminology**. Emphasize the reasons this medicine is a strong option for the end-user. Conclude with a concise summary that bolsters their confidence in making the purchase.

"""

def run_qa(message):
    """
    Run the question-answering model.

    Args:
    message (str): The user's message.
    llm_model (str): The language model to use.

    Returns:
    str: The response from the question-answering model.
    """
    # if not isinstance(message, list):
    #         raise ValueError("'messages' must be a list")
    # for msg in message:
    #     if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
    #         raise ValueError("'messages' must be a list of dictionaries with 'role' and 'content' keys")
    #Finding out the elements in the list
    print("Number of elements in message:", len(message))
    print("Elements in message:")
    for item in message:
        print(item)
    message = filter_messages(message)
    print("Message after removing null values if any..", message)
    # Use memory to store conversation history
    # history = memory.load_memory_variables({}).get('history', '')
    # Always start with the custom prompt
    if not message or message[0].get('role') != 'system':
        system_message = {"role": "system", "content": custom_prompt}
        message.insert(0, system_message)
    # First chat completion with custom prompt for LLM response
    print("message before sending to groq:",message)
    chat_completion = groq_client.chat.completions.create(
        messages=message,
        model="llama3-70b-8192",
        max_tokens=2000,
        temperature=0.5,
        top_p=1,
        stream=False
    )
    response = chat_completion.choices[0].message.content
    logging.info(f"Received response from LLM model: {response}")
    return response
