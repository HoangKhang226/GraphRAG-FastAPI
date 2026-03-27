from agents.researcher import llm
from core.logging_config import logger
import time
import time


def finalize_answer_node(state):
    start_time = time.time()
    logger.info("NODE: Generating final answer...")
    full_history = "\n\n".join(state.context)
    prompt = f"Based on the following research history, provide a comprehensive final answer to: {state.question}\n\nResearch History:\n{full_history}"
    response = llm.invoke(prompt)
    latency = time.time() - start_time
    logger.info(f"NODE: Final answer generated in {latency:.2f}s")
    
    return {"answer": response.content, "history": [response]}
