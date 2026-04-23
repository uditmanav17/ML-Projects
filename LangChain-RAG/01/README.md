## Undersatnding LangChain RAG

- [FutureSmart Blog](https://blog.futuresmart.ai/series/langchain-rag-course)

### **Part 1:**

- **Understanding RAG (Retrieval-Augmented Generation)**
    - What is RAG?
    - Importance of RAG in AI applications
- **Introduction to LangChain and LCEL**
    - Overview of LangChain
    - LangChain Expression Language (LCEL)
- **Exploring LangChain Components**
    - LLM (Large Language Models)
    - Prompts
    - Retrievers
    - Composing components into a chain
- **Document Processing and Vector Databases**
    - Document splitting techniques
    - Embedding documents
    - Storing and retrieving documents in a vector database

- **Creating Your First RAG Chain**
    - Step-by-step guide to building a basic RAG chain
    - Answering questions from documents using RAG
- **Building a Conversational RAG**
    - Handling follow-up questions
    - Concept of contextualizing and refining queries
- **Using pre-built LangChain RAG chains**
    - history_aware_retriever
    - create_retrieval_chain
- **Building Multi User Chatbot**
    - Managing conversation history using a database table

### **Part 2: Moving to Production with FastAPI**

- **Integrating Colab Code with FastAPI**
    - Setting up FastAPI for production
    - Modularizing code into different files for maintainability
    - Implemeting chatbot endpoint to talk to your data
- **Creating API Endpoints**
    - Building endpoints for file upload, list, and deletion
    - Testing and validating endpoints

- **Building a Streamlit Interface**
    - Creating a user-friendly interface with Streamlit
    - Integrating the RAG chatbot API with the Streamlit app
    - File management features (upload, list, delete)