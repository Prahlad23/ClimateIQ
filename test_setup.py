from dotenv import load_dotenv
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import langchain

load_dotenv()

print("Testing ChromaDB...")
client = chromadb.Client()
print("ChromaDB OK")

print("Testing sentence-transformers...")
model = SentenceTransformer("all-MiniLM-L6-v2")
test_vec = model.encode("climate change is accelerating")
print(f"Embedding model OK — vector length: {len(test_vec)}")

print("Testing Groq connection...")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say OK in exactly one word."}]
)
print(f"Groq OK — response: {response.choices[0].message.content}")

print("Testing LangChain...")
print(f"LangChain version: {langchain.__version__}")
print("LangChain OK")

print("\nAll systems ready. You can start building.")

