from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="Enter your question here")
    session_id: Optional[str] = Field("default", description="Session ID for chat history")

class QueryResponse(BaseModel):
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Answer from the Agent")
