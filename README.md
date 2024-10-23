# Langchain LLM Projects
This repository contains various projects that utilize Langchain to demonstrate how Large Language Models (LLMs) can be integrated into different applications such as chatting with websites, Q&A systems, API testing, and more. Each project showcases a unique application of LLMs with distinct use cases.

## Projects Overview
1. **Chat with Websites**
- **Description**: Allows users to input a website URL and interact with the content of the website via an LLM-powered chatbot. The system retrieves information from the website in real-time and enables conversational Q&A.

- **Technologies Used**:
   - Langchain Core
   - Langchain Community
   - Chroma Vectorstore
   - OpenAI API
- File: Chat-with-Websites/src/app.py

2. **Conversational Q&A**
- **Description**: A simple chatbot that allows users to ask questions and receive answers based on a conversational history. The chatbot has a pre-defined role (e.g., comedian assistant) and interacts with users accordingly.
- **Technologies Used**:
    - Langchain
    - Streamlit
    - OpenAI Chat Models
- File: Converational-Q&A/app.py

3. **Demo Q&A**
- **Description**: A sequential chain-based Q&A system that takes a user's input (celebrity name) and generates a detailed response, including biographical information, date of birth, and significant events around that time. The system uses memory to keep track of the conversation.
- **Technologies Used**:
    - Langchain
    - OpenAI API
    - ConversationBufferMemory
    - File: Demo_Q&A/app.py
4. **API Testing**
- **Description**: A FastAPI-based server that exposes endpoints for generating content using Langchain models, including OpenAI's GPT models and Ollama's Llama 2. The client can request an IELTS essay or a poem on any given topic.
- **Technologies Used**:
    - FastAPI
    - Langchain
    - Uvicorn
**Files**:
- Server: api-testing/app.py
- Client: api-testing/client.py

## Features
**Integrations**: Multiple LLMs like OpenAI's GPT-3.5, GPT-4, and Llama 2 via Ollama API.
**Chat History**: Projects like "Chat with Websites" and "Conversational Q&A" maintain a history of user interactions to provide context-aware responses.
**Document Retrieval**: "Chat with Websites" project can retrieve and chat with documents (web content) dynamically.
**API Interaction**: "API Testing" project offers RESTful APIs to interact with LLMs for generating text responses based on user input.

