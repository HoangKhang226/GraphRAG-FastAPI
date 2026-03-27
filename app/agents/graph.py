from typing import List, Optional, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from .researcher import execute_task_node, grade_query, sub_task_node, clarify_question_node
from .writer import finalize_answer_node
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
import operator
from typing import Annotated


class GraphState(BaseModel):
    question: str
    decision: Optional[Literal["ask_back", "split_tasks", "transform"]] = None
    sub_tasks: List[str] = []
    context: List[str] = []
    history: Annotated[List[BaseMessage], operator.add] = []
    answer: str = ""
    is_clear: bool = False


def route_after_grading(state: GraphState):
    if state.decision == "split_tasks":
        return "split"
    elif state.decision == "ask_back":
        return "ask_back"
    else:
        return "process_directly"


def should_continue(state: GraphState):
    return "continue_working" if len(state.sub_tasks) > 0 else "finish_all"


# Initialize Workflow
workflow = StateGraph(GraphState)

workflow.add_node("grade_query", grade_query)
workflow.add_node("sub_task_planning", sub_task_node)
workflow.add_node("execute_task", execute_task_node)
workflow.add_node("clarify_question", clarify_question_node)
workflow.add_node("finalize_answer", finalize_answer_node)

workflow.set_entry_point("grade_query")

workflow.add_conditional_edges(
    "grade_query",
    route_after_grading,
    {"split": "sub_task_planning", "ask_back": "clarify_question", "process_directly": "execute_task"},
)

workflow.add_edge("clarify_question", END)

workflow.add_edge("sub_task_planning", "execute_task")
workflow.add_conditional_edges(
    "execute_task",
    should_continue,
    {"continue_working": "execute_task", "finish_all": "finalize_answer"},
)

# Add Checkpointer for Memory
memory = MemorySaver()

workflow.add_edge("finalize_answer", END)
app = workflow.compile(checkpointer=memory)
