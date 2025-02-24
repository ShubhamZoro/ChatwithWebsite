import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

api_key = st.secrets['api_key']


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']


def get_vectorstore_from_urls(urls):
    all_documents = []

    for url in urls:
        loader = WebBaseLoader(url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(documents)
        all_documents.extend(document_chunks)

    # Create a single Chroma vector store with all document chunks
    vector_store = Chroma.from_documents(all_documents, OpenAIEmbeddings(openai_api_key=api_key))

    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=api_key)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(openai_api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


# App configuration
st.set_page_config(page_title="Chat with Multiple Websites")
st.title("Chat with Multiple Websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_urls = st.text_area("Enter website URLs (comma-separated)").strip()
    load_data = st.button("Load Data")

if load_data and website_urls:
    urls = [url.strip() for url in website_urls.split(',')]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    st.session_state.vector_store = get_vectorstore_from_urls(urls)
    st.success("Data loaded successfully!")

if "vector_store" in st.session_state:
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    # User input
    user_query = st.chat_input("Type your message here--")
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
