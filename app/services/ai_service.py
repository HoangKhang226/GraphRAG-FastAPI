import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Initialize Embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY")
)
