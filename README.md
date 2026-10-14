# HCP Interaction Module

An AI-first CRM assistant for healthcare professional (HCP) interactions, built
with FastAPI and LangGraph. The initial MVP turns interaction notes into a
factual summary and grounded follow-up suggestions.

## Setup

Requires Python 3.11, 3.12, or 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. `OPENAI_MODEL` is optional and defaults to
`gpt-4o-mini`.

## Run

```bash
uvicorn hcp_interaction_module.main:app --reload --app-dir src
```

The interactive API documentation is available at `/docs`; the health endpoint
is `/health`.

## Assist with an interaction

```bash
curl -X POST http://localhost:8000/api/v1/interactions/assist \
  -H "Content-Type: application/json" \
  -d '{
    "hcp_name": "Dr. Example",
    "specialty": "Cardiology",
    "organization": "Example Medical Group",
    "interaction_notes": "Discussed the clinical education materials. The HCP asked us to follow up with available dates for a product demonstration."
  }'
```

The response contains a `summary` and a list of `follow_up_suggestions`. The
LangGraph workflow summarizes first, then generates suggestions from that
summary. Both steps are instructed to use only provided facts and not invent
commitments or clinical recommendations.

## Data handling

This MVP does not store requests or interaction records; it sends the supplied
HCP details and notes to the configured OpenAI model for processing. Avoid
including patient-identifying information or other sensitive data. Review your
organization's privacy, security, and regulatory requirements before using
real interaction data.

## Tests

```bash
pytest
```
