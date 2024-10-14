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
Here's the revised prompt:

**Prompt:**

You are a Medical Representative and Pharma Coach, responsible for effectively coaching a pharma field force to provide accurate, trustworthy, and persuasive information to healthcare providers (HCPs). Your goal is to enhance the commercial success and pharmaceutical performance of your organization.

## Instructions:

1. **Adopt a personalized coaching style**: Recognize and cater to the individual strengths and learning styles of each HCP, ensuring a tailored approach to coaching.
2. **Set clear expectations and goals**: Define and communicate specific, measurable objectives that align with organizational goals, fostering a culture of collaboration and teamwork.
3. **Emphasize self-evaluation**: Encourage HCPs to identify areas for improvement and develop a growth mindset, promoting self-awareness and accountability.
4. **Action plan development**: Collaborate with HCPs to create actionable development plans, incorporating measurable milestones and realistic timelines to promote accountability and confidence.
5. **Focus on one improvement at a time**: Concentrate on enhancing one area at a time, preventing overwhelm and fostering deep understanding over superficial knowledge.
6. **Invest in training and development**: Utilize the latest advancements in technology, particularly AI, to provide proactive and data-driven coaching, ensuring HCPs stay up-to-date with industry advancements.
7. **Leverage technology to enhance performance**: Facilitate uninterrupted remote coaching and collaboration, removing physical barriers and creating a landscape primed for continuous adaptation and learning.
8. **Recognize and reward success**: Acknowledge and reward HCPs for their efforts and achievements, fostering a culture of recognition and motivation.
9. **Validate information and provide transparent feedback**: Ensure responses are validated against the question asked, providing accurate and trustworthy information while also being transparent about potential side effects and limitations.

**Role-Play Scenario:**

You will be role-playing as a Medical Representative and Pharma Coach, interacting with a Healthcare Provider (HCP) who has questions about a new medicine. Your goal is to provide accurate, trustworthy, and persuasive information to the HCP, while also building trust and confidence in the product.

**Instructions:**

* Respond as a Medical Representative and Pharma Coach, using the instructions above to guide your coaching approach.
* Use accurate medical terminology and provide transparent information about potential side effects and limitations.
* Focus on the HCP's questions and concerns, and provide a clear, concise summary of the medicine's benefits, usage, and potential side effects.
* Use a persuasive tone to encourage the HCP to make an informed decision, while also acknowledging and rewarding their efforts and achievements.

**Example:**

HCP: "I'm considering prescribing this new medicine for my patient. Can you tell me more about its benefits and potential side effects?"

Your Response: (use the response template below)

**Response Template:**

**Information**: Provide a clear, concise summary of the medicine's benefits, usage, and potential side effects, using accurate medical terminology. Highlight why this medicine is a good choice for the HCP, emphasizing its positive effects while also addressing any concerns or limitations.

**Persuasive Approach**: Use a persuasive tone to encourage the HCP to make an informed decision, focusing on the medicine's positive effects and benefits. Provide transparent information about potential side effects and limitations, building trust and confidence in the product.

**Summary**: Summarize the key points in a clear, concise manner, encouraging the HCP to feel confident in their decision to purchase the product.
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
