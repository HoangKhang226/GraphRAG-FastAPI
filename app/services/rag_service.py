import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from services.ai_service import embedding, llm
from core.logging_config import logger
from typing import List, Dict
from pydantic import BaseModel, Field

class RAGService:
    def __init__(self):
        self.retriever = None
    
    def process_uploaded_file(self, file_content: bytes, file_name: str):
        try:
            logger.info(f"Processing uploaded file: {file_name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name

            # 2. Load and Split document
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            
            text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            doc_split = text_split.split_documents(docs)

            # 3. Create Vectorstore and update Retriever
            vectorstore = FAISS.from_documents(doc_split, embedding)
            self.retriever = vectorstore.as_retriever()

            # 4. Cleanup: Remove temporary file
            os.remove(temp_path)
            
            logger.info(f"Successfully processed and indexed: {file_name}")
            return f"Successfully processed document: {file_name}"
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {str(e)}")
            return f"Error processing file {file_name}: {str(e)}"
        
    def get_retriever(self):
        return self.retriever

class RelevanceScore(BaseModel):
    index: int = Field(description="The index of the document in the list")
    score: float = Field(description="Relevance score from 0 to 1")

class RerankResult(BaseModel):
    scores: List[RelevanceScore]

    def search_with_rerank(self, query: str, top_k: int = 5, top_n: int = 3) -> List[str]:
        """
        Retrieves top_k documents and reranks them using LLM to return top_n most relevant ones.
        """
        if not self.retriever:
            return []
        
        # 1. Base Retrieval
        logger.info(f"Retrieving {top_k} documents for query: {query}")
        docs = self.retriever.invoke(query)
        if not docs:
            return []
            
        doc_contents = [d.page_content for d in docs]
        
        # 2. LLM Reranking
        try:
            logger.info(f"Reranking documents using LLM...")
            
            # Format docs for the prompt
            formatted_docs = "\n".join([f"[{i}] {doc}" for i, doc in enumerate(doc_contents)])
            
            rerank_prompt = f"""You are an Information Retrieval Expert.
            Rate the relevance of the following document chunks to the user query.
            Return a list of scores between 0 and 1 for each chunk.
            
            Query: {query}
            
            Chunks:
            {formatted_docs}
            """
            
            structured_llm = llm.with_structured_output(RerankResult)
            rerank_output = structured_llm.invoke(rerank_prompt)
            
            # Map scores back to documents
            scored_docs = []
            for item in rerank_output.scores:
                if 0 <= item.index < len(doc_contents):
                    scored_docs.append((doc_contents[item.index], item.score))
            
            # Sort by score and take top_n
            final_scored = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            return [doc for doc, score in final_scored[:top_n]]
            
        except Exception as e:
            logger.warning(f"LLM Reranking failed: {str(e)}. Falling back to base retrieval.")
            return doc_contents[:top_n]
    
rag_manager = RAGService()