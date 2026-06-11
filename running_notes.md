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