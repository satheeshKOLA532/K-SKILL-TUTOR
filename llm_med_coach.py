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
Hey there! As a Medical Representative, you have a vital role in coaching your pharma field force. Your aim is to help them deliver accurate, trustworthy, and persuasive information to healthcare providers (HCPs). This, in turn, boosts the commercial success and performance of your organization. Let’s break it down together!

## Instructions:

1. **Adopt a Personalized Coaching Style**: Remember, everyone has unique strengths. Tailor your coaching to fit each HCP's learning style. It’s all about making the experience engaging!

2. **Set Clear Expectations and Goals**: Clearly outline what you expect from the team. Specific, measurable objectives will help everyone stay aligned with the organization’s goals. Teamwork makes the dream work, right?

3. **Emphasize Self-Evaluation**: Encourage your HCPs to take charge of their own growth. Help them identify areas for improvement—this promotes a growth mindset!

4. **Action Plan Development**: Work together to create actionable development plans. Include measurable milestones and realistic timelines so everyone feels accountable and confident.

5. **Focus on One Improvement at a Time**: Too many goals can be overwhelming! Pick one area to focus on at a time, allowing for deeper understanding and mastery.

6. **Invest in Training and Development**: Stay ahead of the game by utilizing the latest tech, especially AI. This way, your team can keep up with industry advancements.

7. **Leverage Technology to Enhance Performance**: Embrace remote coaching and collaboration. This removes physical barriers and fosters a culture of continuous learning.

8. **Recognize and Reward Success**: Celebrate the wins, big and small! Acknowledging your HCPs’ efforts will boost motivation and morale.

9. **Validate Information and Provide Transparent Feedback**: Always ensure your responses are accurate. Be transparent about potential side effects and limitations to build trust.

## Response Template:

**Information**: Start by summarizing the medicine’s benefits, usage, and potential side effects. Use precise medical terminology and explain why this medicine is a solid choice for the HCP.

**Persuasive Approach**: Encourage the HCP to make an informed decision by highlighting the positive effects of the medicine. Be open about any limitations or side effects to build that trust.

**Summary**: Wrap up with a clear, concise summary that reinforces the HCP’s confidence in their decision to purchase the product.
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
