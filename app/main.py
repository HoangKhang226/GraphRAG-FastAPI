import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from schemas.request_schema import QueryRequest, QueryResponse
from agents.graph import app as agent_app
from services.rag_service import rag_manager
from services.cache_service import semantic_cache
from core.logging_config import logger

app = FastAPI(title="Multi-Agent RAG System")

@app.post("/upload", summary="Upload PDF document", description="Upload a PDF file for the Agent to learn new information.")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    content = await file.read()
    message = rag_manager.process_uploaded_file(content, file.filename)
    return {"message": message}

@app.post("/query", response_model=QueryResponse, summary="Q&A with AI Agent", description="Ask a question. Use session_id to maintain chat history.")
async def handle_query(question: str, session_id: Optional[str] = "default"):
    logger.info(f"Received query: {question} (Session: {session_id})")
    
    # Check Semantic Cache
    cached_answer = semantic_cache.get(question)
    if cached_answer and len(cached_answer) > 0:
        logger.info("Serving answer from Semantic Cache. Syncing history...")
        # Sync history to LangGraph even on cache hit
        config = {"configurable": {"thread_id": session_id}}
        agent_app.update_state(config, {"history": [HumanMessage(content=question), AIMessage(content=cached_answer)]})
        return QueryResponse(question=question, answer=cached_answer)

    try:
        # Configuration for Memory (thread_id)
        config = {"configurable": {"thread_id": session_id}}
        
        # Initial state
        initial_state = {
            "question": question,
            "sub_tasks": [],
            "context": [],
            "history": [],
            "answer": "",
            "is_clear": False
        }
        
        # Invoke the graph with history
        logger.info("Invoking agent workflow with session context...")
        result = agent_app.invoke(initial_state, config=config)
        logger.info("Workflow completed successfully.")
        
        # Save to Cache
        semantic_cache.set(question, result.get("answer", ""))
        
        return QueryResponse(
            question=question,
            answer=result.get("answer", "No answer generated.")
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
