from typing import Protocol, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class InteractionSummary(BaseModel):
    summary: str = Field(description="A concise, factual summary of the interaction.")


class FollowUpPlan(BaseModel):
    suggestions: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Actionable proposed follow-up tasks grounded in the interaction.",
    )


class AssistantState(TypedDict, total=False):
    hcp_name: str
    specialty: str
    organization: str | None
    interaction_notes: str
    summary: str
    follow_up_suggestions: list[str]


class StructuredOutputModel(Protocol):
    def with_structured_output(self, schema: type[BaseModel]) -> Runnable: ...


def build_assistant_graph(model: StructuredOutputModel):
    def summarize_interaction(state: AssistantState) -> dict[str, str]:
        organization = state.get("organization") or "Not provided"
        response = model.with_structured_output(InteractionSummary).invoke(
            [
                SystemMessage(
                    content=(
                        "Summarize the HCP interaction concisely and factually. "
                        "Use only information in the notes; do not infer outcomes, "
                        "commitments, or clinical recommendations."
                    )
                ),
                HumanMessage(
                    content=(
                        f"HCP: {state['hcp_name']}\n"
                        f"Specialty: {state['specialty']}\n"
                        f"Organization: {organization}\n"
                        f"Interaction notes:\n{state['interaction_notes']}"
                    )
                ),
            ]
        )
        return {"summary": response.summary}

    def suggest_follow_ups(state: AssistantState) -> dict[str, list[str]]:
        response = model.with_structured_output(FollowUpPlan).invoke(
            [
                SystemMessage(
                    content=(
                        "Suggest practical next steps for a CRM user based only on the "
                        "interaction summary. Do not invent commitments, owners, dates, "
                        "clinical advice, or facts. Phrase unsupported details as "
                        "questions to confirm."
                    )
                ),
                HumanMessage(
                    content=(
                        f"HCP: {state['hcp_name']} ({state['specialty']})\n"
                        f"Interaction summary:\n{state['summary']}"
                    )
                ),
            ]
        )
        return {"follow_up_suggestions": response.suggestions}

    workflow = StateGraph(AssistantState)
    workflow.add_node("summarize_interaction", summarize_interaction)
    workflow.add_node("suggest_follow_ups", suggest_follow_ups)
    workflow.add_edge(START, "summarize_interaction")
    workflow.add_edge("summarize_interaction", "suggest_follow_ups")
    workflow.add_edge("suggest_follow_ups", END)
    return workflow.compile()
