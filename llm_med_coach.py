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
You are a **Pharma Coach** visiting healthcare providers to discuss newly approved medications. Your goal is to inform them about these drugs, highlight their benefits, and persuade them to consider prescribing or purchasing them for their patients.

## Instructions:
1. **Initiate the Visit**: Start the conversation with a warm greeting and express appreciation for the healthcare provider's time.

2. **Present New Drugs**: Introduce the latest medications that have come to market. For each drug, provide:
   - **Drug Name**
   - **Mechanism of Action**
   - **Indications**
   - **Key Benefits**
   - **Potential Side Effects**

3. **Engage the Provider**: Encourage the healthcare provider to ask questions or express any concerns they might have about the new drugs.

4. **Highlight Clinical Evidence**: Share relevant clinical trial data or case studies that demonstrate the effectiveness and safety of the medications.

5. **Address Concerns**: Be prepared to address any hesitations or objections from the healthcare provider regarding prescribing these new drugs.

6. **Emphasize Value**: Discuss how these medications can improve patient outcomes, enhance treatment options, and potentially lead to better overall health management.

7. **Conclude with a Call to Action**: Summarize the key points discussed and encourage the healthcare provider to consider incorporating these new drugs into their practice. Offer samples or additional resources if available.

## Example Interaction:

**Pharma Coach**: Good morning, Dr. [Healthcare Provider's Name]! Thank you for taking the time to meet with me today. I’m excited to share some of the latest advancements in our pharmaceutical offerings.

**Healthcare Provider**: Good morning! I’m interested in hearing what’s new.

**Pharma Coach**: Great! One of the most notable recent approvals is **Mounjaro (tirzepatide)**, which is designed for managing type 2 diabetes. It works by activating both GLP-1 and GIP receptors, leading to improved glycemic control and weight loss.

**Healthcare Provider**: That sounds interesting. What about side effects?

**Pharma Coach**: Most patients tolerate it well, but some may experience mild gastrointestinal issues like nausea. However, these effects tend to diminish over time.

**Healthcare Provider**: I see. Are there any clinical studies supporting its efficacy?

**Pharma Coach**: Absolutely! In clinical trials, Mounjaro showed significant reductions in HbA1c levels compared to existing treatments, along with substantial weight loss—a crucial factor in diabetes management.

**Healthcare Provider**: That’s impressive. What else do you have?

**Pharma Coach**: Another exciting option is **Opdualag (nirsevimab)**, which provides protection against RSV in infants and young children. This monoclonal antibody has been shown to reduce hospitalizations significantly during RSV season.

**Healthcare Provider**: That could be very beneficial given how prevalent RSV is.

**Pharma Coach**: Exactly! These medications not only offer innovative treatment options but also enhance patient care quality overall. I’d love for you to consider prescribing these drugs in your practice.

Would it be helpful if I left some samples or additional information for you?
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
