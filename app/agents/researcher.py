import os
from typing import List, Optional, Literal, Annotated
import time
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from services.ai_service import llm
from services.rag_service import rag_manager
from core.logging_config import logger

# llm and retriever are imported from services

# --- TOOLS DEFINITION ---

def web_search_safe(query: str):
    try:
        tool = TavilySearchResults(k=3)
        return tool.invoke({"query": query})
    except Exception as e:
        return f"Web Search Error: {str(e)}. Try refining your web query."


web_search_tool = Tool(
    name="tavily_web_search",
    description="Search the web for real-time information. Useful for external news, events, or facts not in datastore.",
    func=web_search_safe,
)


def rag_search_safe(query: str):
    try:
        retriever = rag_manager.get_retriever()
        if retriever is None:
            return "RAG Error: No documents have been uploaded for search yet."
        # Use the reranked search instead of raw retriever
        docs = rag_manager.search_with_rerank(query, top_k=5, top_n=3)
        return "\n".join(docs) if docs else "No relevant information found in documents."
    except Exception as e:
        return f"RAG Error: {str(e)}."


rag_tool = Tool(
    name="internal_knowledge_search",
    description="Searches and returns internal company rules, policies, and business logic mapping.",
    func=rag_search_safe,
)

tools = [rag_tool, web_search_tool]

# --- SCHEMAS ---


class GradeResult(BaseModel):
    """Schema for evaluating the user's question."""

    is_clear: bool = Field(description="Whether the question is clear or not")
    decision: Literal["ask_back", "split_tasks", "transform"] = Field(
        description="Routing decision based on the question's content"
    )


class TaskList(BaseModel):
    """A list of sequential steps to fulfill a user request."""

    tasks: List[str] = Field(
        description="A list of clear, executable steps in logical order"
    )

# --- NODES ---


def grade_query(state):
    """
    Evaluates the user's question to determine the next workflow step.
    """
    logger.info(f"NODE: Grading query: {state.question}")
    grade_prompt = """
    You are a query router.

    Analyze the user question and decide:
    1. Whether the question is clear.
    2. What action to take next.

    Routing rules:
    - ask_back: unclear, vague, or missing intent
    - split_tasks: multiple steps or compound request
    - transform: clear but needs rewriting for retrieval/search

    Return your answer strictly following the provided schema.
    """
    structured_llm = llm.with_structured_output(GradeResult)

    prompt = ChatPromptTemplate.from_messages(
        [("system", grade_prompt), ("human", "User question: {question}")]
    )

    chain = prompt | structured_llm
    # Include history in the call
    messages = [SystemMessage(content=grade_prompt)] + state.history + [HumanMessage(content=f"User question: {state.question}")]
    result: GradeResult = structured_llm.invoke(messages)

    # Persist the user's question at the start of the workflow if history is empty or last msg is AI
    # (Actually, easier to just add it here for the first node)
    new_history = [HumanMessage(content=state.question)]
    
    return {"is_clear": result.is_clear, "decision": result.decision, "history": new_history}


def sub_task_node(state):
    """
    Decomposes a complex user request into smaller, sequential sub-tasks.
    """
    logger.info("NODE: Sub-task planning...")
    planner_prompt = """You are an expert Strategic Planner. 
    Your goal is to break down a complex 'User Question' into a list of smaller, logical, and executable sub-tasks.
    
    Rules:
    - Breakdown: Each task must be a single, independent action.
    - Sequence: Arrange tasks in a logical chronological order."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", planner_prompt), ("human", "User Question: {question}")]
    )
    planner_chain = prompt | llm.with_structured_output(TaskList)

    # Include history in the call
    messages = [SystemMessage(content=planner_prompt)] + state.history + [HumanMessage(content=f"User Question: {state.question}")]
    result = llm.with_structured_output(TaskList).invoke(messages)

    return {"sub_tasks": result.tasks}


def clarify_question_node(state):
    """
    Generates a clarification request when the user's question is unclear.
    """
    logger.info("NODE: Generating clarification request...")
    prompt = f"The user asked: '{state.question}'. This is unclear in the current context. Ask a polite follow-up question to clarify what they need."
    response = llm.invoke(state.history + [HumanMessage(content=prompt)])
    
    return {"answer": response.content, "history": [response]}


def execute_task_node(state):
    """
    Executes the first sub-task by manually routing tool calls from the LLM.
    """
    start_time = time.time()
    if not state.sub_tasks:
        current_task = state.question
    else:
        current_task = state.sub_tasks[0]
    
    logger.info(f"NODE: Executing task: {current_task}")

    llm_with_tools = llm.bind_tools(tools)
    existing_context = (
        "\n".join(state.context) if state.context else "No prior research context."
    )
    
    # SYSTEM PROMPT with Memory instructions
    system_msg = SystemMessage(
        content="""You are a professional AI Researcher. 
        Solve the 'Current Task' using tools. 
        You have access to 'internal_knowledge_search' (RAG) and 'tavily_web_search'.
        
        Use the 'Context' and 'History' to avoid redundant work.
        Always provide clear, factual results."""
    )
    
    # Combine History and Current Task
    messages = [system_msg] + state.history + [
        HumanMessage(content=f"Context from current session:\n{existing_context}\n\nTask: {current_task}")
    ]

    response = llm_with_tools.invoke(messages)

    step_output = ""

    if response.tool_calls:
        for tool_call in response.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            logger.info(f"Tool Call: {t_name} with args: {t_args}")

            # Extract query with multiple fallbacks
            query_val = t_args.get("query") or t_args.get("quey") or t_args.get("input")
            
            # If still None, try getting the first value from the dict or fallback to current task/question
            if not query_val and t_args:
                query_val = list(t_args.values())[0]
            
            if not query_val:
                query_val = current_task or state.question
                logger.warning(f"Tool {t_name} had no query in args. Falling back to: {query_val}")

            if t_name == "internal_knowledge_search":
                logger.info(f"-> Running RAG with keyword: {query_val}")
                result = rag_tool.func(query=query_val)
                step_output += f"\n[RAG Result]: {result}"

            elif t_name == "tavily_web_search":
                logger.info(f"-> Running Web Search with keyword: {query_val}")
                result = web_search_tool.func(query=str(query_val))
                step_output += f"\n[Web Result]: {result}"
    else:
        step_output = response.content

    final_context_entry = f"Task: {current_task}\nResult: {step_output}"

    latency = time.time() - start_time
    logger.info(f"NODE: Task executed in {latency:.2f}s")

    return {
        "context": state.context + [final_context_entry],
        "sub_tasks": state.sub_tasks[1:],
    }
