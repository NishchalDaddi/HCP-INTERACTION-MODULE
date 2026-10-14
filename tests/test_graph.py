from langchain_core.messages import HumanMessage

from hcp_interaction_module.graph import (
    FollowUpPlan,
    InteractionSummary,
    build_assistant_graph,
)


class StubStructuredOutput:
    def __init__(self, response):
        self.response = response
        self.inputs = []

    def invoke(self, messages):
        self.inputs.append(messages)
        return self.response


class StubModel:
    def __init__(self):
        self.summary = StubStructuredOutput(
            InteractionSummary(summary="Discussed a product demonstration.")
        )
        self.follow_ups = StubStructuredOutput(
            FollowUpPlan(suggestions=["Confirm a date for the demonstration."])
        )

    def with_structured_output(self, schema):
        if schema is InteractionSummary:
            return self.summary
        if schema is FollowUpPlan:
            return self.follow_ups
        raise AssertionError(f"Unexpected structured output schema: {schema}")


def test_graph_summarizes_then_generates_follow_ups() -> None:
    model = StubModel()
    graph = build_assistant_graph(model)

    result = graph.invoke(
        {
            "hcp_name": "Dr. Example",
            "specialty": "Cardiology",
            "organization": None,
            "interaction_notes": "The HCP asked about a product demonstration.",
        }
    )

    assert result["summary"] == "Discussed a product demonstration."
    assert result["follow_up_suggestions"] == [
        "Confirm a date for the demonstration."
    ]
    summary_input = model.summary.inputs[0]
    assert any(
        isinstance(message, HumanMessage)
        and "The HCP asked about a product demonstration." in message.content
        for message in summary_input
    )
    follow_up_input = model.follow_ups.inputs[0]
    assert any(
        isinstance(message, HumanMessage)
        and "Discussed a product demonstration." in message.content
        for message in follow_up_input
    )
