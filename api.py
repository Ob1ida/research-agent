"""
FastAPI wrapper for the research agent.

    pip install fastapi uvicorn
    uvicorn api:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import agent

app = FastAPI(
    title="Research Agent API",
    description="A web-search agent built with LangGraph and Claude.",
    version="1.0.0",
)


# --- Request and response shapes ---

class AskRequest(BaseModel):
    question: str = Field(description="What to ask the agent.")
    session_id: str = Field(
        default="default",
        description="Conversations with the same session_id share memory.",
    )


class ToolCall(BaseModel):
    name: str
    args: dict


class AskResponse(BaseModel):
    answer: str
    session_id: str
    tool_calls: list[ToolCall] = Field(
        description="Which tools the agent used to answer."
    )
    steps: int = Field(description="How many messages the agent produced.")


# --- Endpoints ---

@app.get("/health")
def health():
    """Check that the server is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Send a question to the agent and get an answer back."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    config = {
        "configurable": {"thread_id": req.session_id},
        "recursion_limit": 15,
    }

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": req.question}]},
            config=config,
        )
    except Exception as e:
        print(f"Agent failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="The agent failed to answer.")

    messages = result["messages"]

    # Collect the tools used, so the client can see how the answer was made.
    tool_calls = [
        ToolCall(name=tc["name"], args=tc["args"])
        for m in messages
        if getattr(m, "tool_calls", None)
        for tc in m.tool_calls
    ]

    return AskResponse(
        answer=messages[-1].content,
        session_id=req.session_id,
        tool_calls=tool_calls,
        steps=len(messages),
    )


@app.delete("/session/{session_id}")
def reset_session(session_id: str):
    """Forget one conversation. The next question starts fresh."""
    return {
        "session_id": session_id,
        "note": "Use a different session_id to start fresh.",
    }