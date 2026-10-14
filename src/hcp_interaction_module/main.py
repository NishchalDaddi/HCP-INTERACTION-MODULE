import asyncio
import logging
from functools import partial

from fastapi import FastAPI, HTTPException, status
from langchain_google_genai import ChatGoogleGenerativeAI

from hcp_interaction_module.config import get_settings 
settings = get_settings()
from hcp_interaction_module.graph import build_assistant_graph
from hcp_interaction_module.schemas import (
    InteractionAssistRequest,
    InteractionAssistResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HCP Interaction Assistant",
    description="Summarize HCP interactions and suggest grounded CRM follow-ups.",
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


def _run_assistant(request: InteractionAssistRequest) -> InteractionAssistResponse:
    try:
        model = ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=settings.google_model,
            temperature=0,
        )
        graph = build_assistant_graph(model)
        result = graph.invoke(request.model_dump())
        print(f"DEBUG - GRAPH RESULT: {result}")

        if not result or not isinstance(result, dict):
            raise ValueError(f"Graph returned non-dictionary result: {result}")

        return InteractionAssistResponse(
            summary=result.get("summary", "No summary generated"),
            follow_up_suggestions=result.get("follow_up_suggestions", []),
        )
    except Exception as e:
        print(f"CRITICAL ERROR IN ASSISTANT: {e}")
        import traceback
        traceback.print_exc()
        raise e

def _run_assistant(request: InteractionAssistRequest) -> InteractionAssistResponse:
    model = ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model=settings.google_model,
        temperature=0,
    )
    result = build_assistant_graph(model).invoke(request.model_dump())
    return InteractionAssistResponse(
        summary=result["summary"],
        follow_up_suggestions=result["follow_up_suggestions"],
    )


@app.post(
    "/api/v1/interactions/assist",
    response_model=InteractionAssistResponse,
)
async def assist_with_interaction(
    request: InteractionAssistRequest,
) -> InteractionAssistResponse:
    if not get_settings().google_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The AI assistant is not configured. Set GOOGLE_API_KEY.",
        )
    return _run_assistant(request)


@app.get("/health")
async def health_check():
    return {"status": "ok"}