from pydantic import BaseModel, Field


class InteractionAssistRequest(BaseModel):
    hcp_name: str = Field(min_length=1, max_length=200)
    specialty: str = Field(min_length=1, max_length=200)
    organization: str | None = Field(default=None, max_length=200)
    interaction_notes: str = Field(min_length=1, max_length=20_000)


class InteractionAssistResponse(BaseModel):
    summary: str
    follow_up_suggestions: list[str]
