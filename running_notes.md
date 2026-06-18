## 2 June 2026

### What I did
- Completed full environment setup
- Installed all libraries successfully
- Tested ChromaDB, sentence-transformers, Groq, and LangChain

### What is working
- ChromaDB OK
- Embedding model: all-MiniLM-L6-v2, vector length 384
- Groq connection OK, model: llama-3.1-8b-instant
- LangChain version 1.3.2

### Notes
- Old Groq model llama3-8b-8192 is retired, use llama-3.1-8b-instant
- Libraries were initially installed outside venv, had to reinstall inside venv

### Next session
- Open Jupyter notebook and start parsing IPCC PDFs

## 7 June 2026

### What I did
- Completed Phase 1 — document parsing and cleaning
- Parsed all 6 IPCC PDFs using pdfplumber
- Extracted and cleaned text from all documents
- Saved cleaned txt files to Data/processed/
- Understood every line of the parsing code in detail

### Stats
- WG1 SPM: 32 pages, ~24,557 words
- WG1 TS: 112 pages, ~98,198 words
- WG2 SPM: 34 pages, ~32,925 words
- WG2 TS: 84 pages, ~86,696 words
- WG3 SPM: 56 pages, ~44,432 words
- WG3 TS: 102 pages, ~90,173 words
- Total: 420 pages, ~377,000 words

### What I learned
- How pdfplumber extracts text page by page
- How clean_text removes noise like page numbers and extra blank lines
- How generator expressions work in Python
- How try/except handles errors without crashing
- How os.makedirs creates folders automatically
- How f-strings embed variables inside printed text
- Decision made to switch to LangChain for the rest of the project
- Fixed VS Code kernel issue — always launch via cmd terminal with venv activated

### Key decisions
- Switching from raw pdfplumber to LangChain going forward
- LangChain is industry standard and better for CV
- Keeping 01_document_parsing.ipynb as proof of understanding underlying concepts

### Next session (11 June)
- Start 02_langchain_pipeline.ipynb
- Load single PDF using PyPDFLoader
- Then load all PDFs using DirectoryLoader with lazy_load
- Then move to chunking with RecursiveCharacterTextSplitter

## 11 June 2026

### What I did
- Started 02_langchain_pipeline.ipynb
- Loaded single PDF using PyPDFLoader
- Loaded all 6 PDFs using DirectoryLoader with lazy_load and multithreading
- Applied basic text cleaning to all documents
- Understood LangChain Document object structure — page_content and metadata

### What I learned
- PyPDFLoader loads one PDF, DirectoryLoader loads entire folders
- Each loaded page is a LangChain Document object with page_content and metadata
- lazy_load handles memory efficiently — one document at a time
- use_multithreading speeds up loading by processing multiple PDFs in parallel
- defaultdict automatically initialises new keys with a default value
- split('\\')[-1] extracts just the filename from a full file path
- Cleaning is optional for good quality digital PDFs like IPCC documents

### Key decisions
- Skipping cleaning step for now — IPCC PDFs are clean enough
- Will revisit cleaning only if retrieval quality suffers during evaluation

### Stats confirmed
- Total pages loaded: 420 across 6 documents
- WG1 SPM: 32, WG1 TS: 112, WG2 SPM: 34, WG2 TS: 84, WG3 SPM: 56, WG3 TS: 102

### Next session
- Chunking using RecursiveCharacterTextSplitter
- Test three chunk sizes: 200, 500, 800 words
- Understand what chunks look like before moving to embedding

## 16 June 2026

### What I did
- Created three ChromaDB collections for chunk sizes 1000, 2500, 4000
- Embedded all 3918 chunks using all-MiniLM-L6-v2
- Connected Groq LLM (llama-3.1-8b-instant) to retriever
- Built complete end to end RAG pipeline using modern LangChain LCEL syntax
- Tested pipeline with multiple questions

### Key findings
- Pipeline works correctly for specific factual questions
- "What is the current global temperature rise?" — answered correctly, cited A.1.2 WG1 SPM
- "What will happen to sea levels by 2100?" — answered correctly
- "What are the main causes of climate change?" — FAILED (retrieval miss)
- Rephrasing to "human influence greenhouse gas emissions" — succeeded immediately
- RESEARCH FINDING: Dense retrieval fails when user vocabulary differs from document vocabulary
- This confirms why hybrid retrieval with BM25 is needed

### What I learned
- RetrievalQA is deprecated in modern LangChain — use LCEL chain instead
- Modern RAG chain uses pipe operator | to connect components
- Retrieval failures are content problems not system problems
- Same information retrieved successfully with different query phrasing

### Next session
- Add source document display to answers
- Build BM25 retriever
- Compare dense vs BM25 on same failed question

## 16-18 June 2026 — Retrieval Investigation and Findings

### The Query We Investigated
"What are the main causes of climate change?"

### What We Confirmed
The correct answer exists in the database — WG1 SPM Page 3:
"Observed increases in well-mixed greenhouse gas (GHG) concentrations 
since around 1750 are unequivocally caused by human activities."

This chunk is clean, properly stored, and retrievable with technical vocabulary.

### What We Tried and Results

1. DENSE RETRIEVAL — FAILED
   Retrieved chunks about displacement, coral bleaching, heat mortality.
   Reason: Semantic vectors for "main causes of climate change" and 
   "unequivocally caused by human activities" are not close enough 
   because surface vocabulary is too different despite similar meaning.

2. BM25 RETRIEVAL — FAILED
   Retrieved chunks about mitigation pathways and model methodologies.
   Reason: Word "causes" matched to "caused" but BM25 was confused by 
   high frequency of climate-related terms across all documents. Term 
   frequency noise overwhelmed the keyword signal.

3. HYBRID RETRIEVAL (EnsembleRetriever) — FAILED
   Retrieved 5 chunks, partially relevant but still missed A.1.1.
   Reason: Both underlying methods failed so combining them could not 
   overcome the fundamental vocabulary gap. Result 2 contained 
   "Human-induced climate change" which was partially relevant.

4. MULTI QUERY RETRIEVER — FAILED
   Retrieved 10 chunks across multiple query variations, still missed A.1.1.
   Reason: LLM generated alternative phrasings but none were technical 
   enough to bridge the vocabulary gap to the exact chunk.

5. DIRECT TECHNICAL VOCABULARY SEARCH — SUCCESS
   Query: "greenhouse gas concentrations caused human activities"
   Retrieved A.1.1 as Result 1 immediately.
   Reason: Query vocabulary matched document vocabulary closely enough 
   for semantic similarity to work correctly.

### Root Cause Analysis
This is a vocabulary mismatch problem — a documented and known 
limitation of RAG systems on scientific literature. Confirmed by 
published research (KohakuRAG 2025, RAG taxonomy surveys 2024).

The IPCC uses highly specialised technical vocabulary. Non-expert users 
ask questions in everyday language. The semantic gap between these two 
registers is too large for any of the four retrieval methods to bridge 
without additional preprocessing.

### What Helped Partially
- Header cleaning (removing Box 9.1, Figure SPM.1, A.1.1 labels) 
  improved chunk quality. Dense retrieval results became more relevant 
  after cleaning even if A.1.1 was still not retrieved.
- Result quality improved from completely irrelevant to partially 
  relevant after cleaning.

### LangChain Import Issues Encountered
- EnsembleRetriever not in langchain.retrievers or langchain_community
- Root cause: LangChain 1.0 (November 2025) moved legacy retrievers 
  to langchain_classic package
- Fix: pip install langchain-classic, then 
  from langchain_classic.retrievers import EnsembleRetriever
- MultiQueryRetriever found in langchain_community after searching
- Rule going forward: langchain_core for abstractions, 
  langchain_community for loaders/BM25, langchain_classic for legacy 
  retrievers, langchain_groq/chroma/huggingface for providers

### Research Implication
All four retrieval strategies fail on conceptual natural language 
queries over highly technical scientific literature. This is the 
central finding of the retrieval comparison phase and will form the 
core of the research report's analysis section.

Query transformation or domain-specific query expansion is needed 
as an additional layer — this is the next step to implement.

### Next Steps
- Implement simple query expansion using LLM before retrieval
- Build complete RAG chain connecting all retrieval methods
- Start benchmark dataset construction
- Run formal experiments across all configurations