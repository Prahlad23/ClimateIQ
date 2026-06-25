import pickle
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv
import os
import re

import streamlit as st
import os

# Get API key
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    from dotenv import load_dotenv
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")

# Page config
st.set_page_config(
    page_title="ClimateIQ",
    page_icon="🌍",
    layout="centered"
)

# Title
st.title("🌍 ClimateIQ")
st.subheader("Ask anything about climate change — powered by IPCC AR6")
st.markdown("---")

# Cache expensive resources so they load only once
@st.cache_resource
def load_resources():
    
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    chroma_path = os.path.join(os.path.dirname(__file__), "Data", "chroma_db")
    
    # Load vectorstore
    vectorstore = Chroma(
        collection_name="climate_chunks_2500",
        embedding_function=embeddings,
        persist_directory=chroma_path
    )
    
    # Load pre-saved chunks for BM25
    chunks_path = os.path.join(os.path.dirname(__file__), "chunks.pkl")
    with open(chunks_path, "rb") as f:
        all_chunk_sets = pickle.load(f)
    chunks = all_chunk_sets[2500]
    
    # Build BM25 index
    bm25_retriever = BM25Retriever.from_documents(chunks, k=3)
    
    # Load LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_key,
        temperature=0
    )
    
    return embeddings, vectorstore, bm25_retriever, chunks, llm, chroma_path

# Load everything
with st.spinner("Loading ClimateIQ..."):
    embeddings, vectorstore, bm25_retriever, chunks, llm, chroma_path = load_resources()

# RAG prompt
rag_prompt = PromptTemplate.from_template("""You are ClimateIQ, a climate science assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."
Always mention which document your answer comes from.
If the question is simple, answer in plain language.
If the question is technical, use appropriate scientific language.

Context:
{context}

Question: {question}

Answer:""")

# Query expansion prompt
expansion_prompt = PromptTemplate.from_template("""You are a climate science expert.
Rewrite this question using precise scientific terminology from IPCC reports.
Return only the rewritten question.

Original question: {question}

Rewritten question:""")

query_expander = expansion_prompt | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_pipeline(query, retriever_type):
    
    if retriever_type == "Dense":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        search_query = query
        
    elif retriever_type == "BM25":
        retriever = bm25_retriever
        search_query = query
        
    elif retriever_type == "Hybrid":
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[0.5, 0.5]
        )
        search_query = query
        
    elif retriever_type == "Dense + Query Expansion":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        search_query = query_expander.invoke({"question": query})
    
    retrieved_docs = retriever.invoke(search_query)
    context = format_docs(retrieved_docs)
    final_prompt = rag_prompt.format(context=context, question=query)
    response = llm.invoke(final_prompt)
    
    return response.content, retrieved_docs, search_query

# Sidebar
with st.sidebar:
    st.header("Settings")
    retriever_type = st.selectbox(
        "Retrieval Method",
        ["Hybrid", "Dense", "BM25", "Dense + Query Expansion"],
        index=0
    )
    st.markdown("---")
    st.markdown("**About ClimateIQ**")
    st.markdown("Built on IPCC AR6 reports using RAG")
    st.markdown("University of Oulu Research Internship")

# Main interface
query = st.text_input(
    "Ask a climate science question:",
    placeholder="e.g. What is the current global temperature rise?"
)

if query:
    with st.spinner("Searching climate science literature..."):
        answer, retrieved_docs, search_query = run_pipeline(query, retriever_type)
    
    # Show answer
    st.markdown("### Answer")
    st.markdown(answer)
    
    # Show sources
    st.markdown("### Sources")
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "").split("\\")[-1]
        page = doc.metadata.get("page", "")
        with st.expander(f"Source {i+1} — {source}, Page {page}"):
            st.markdown(doc.page_content)
    
    # Show expanded query if used
    if retriever_type == "Dense + Query Expansion" and search_query != query:
        with st.expander("Query after expansion"):
            st.markdown(f"*{search_query}*")