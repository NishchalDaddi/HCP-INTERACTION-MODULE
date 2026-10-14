from fastapi.testclient import TestClient

from hcp_interaction_module import main
from hcp_interaction_module.schemas import InteractionAssistResponse


client = TestClient(main.app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_assist_requires_openai_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: type("Settings", (), {"openai_api_key": None})(),
    )

    response = client.post(
        "/api/v1/interactions/assist",
        json={
            "hcp_name": "Dr. Example",
            "specialty": "Cardiology",
            "interaction_notes": "Discussed a follow-up product demonstration.",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "The AI assistant is not configured. Set OPENAI_API_KEY."
    )


def test_assist_returns_summary_and_follow_ups(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "get_settings",
        lambda: type("Settings", (), {"openai_api_key": "configured"})(),
    )
    monkeypatch.setattr(
        main,
        "_run_assistant",
        lambda request: InteractionAssistResponse(
            summary="Discussed a product demonstration.",
            follow_up_suggestions=["Confirm a date for the demonstration."],
        ),
    )

    response = client.post(
        "/api/v1/interactions/assist",
        json={
            "hcp_name": "Dr. Example",
            "specialty": "Cardiology",
            "interaction_notes": "Discussed a follow-up product demonstration.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "Discussed a product demonstration.",
        "follow_up_suggestions": ["Confirm a date for the demonstration."],
    }


def test_assist_validates_interaction_input() -> None:
    response = client.post(
        "/api/v1/interactions/assist",
        json={
            "hcp_name": "",
            "specialty": "Cardiology",
            "interaction_notes": "Discussed a follow-up.",
        },
    )

    assert response.status_code == 422
