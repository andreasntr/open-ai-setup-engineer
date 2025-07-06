from typing import Literal
from pydantic import BaseModel, Field


class ReasoningQA(BaseModel):
    question: str
    reasoning: str
    answer: Literal['yes', 'no']


class CornerDetails(BaseModel):
    reference: str = Field(
        description="verbatim name of the corner as input by the user"
    )
    corner: str = Field(
        description="description of the matching corner, verbatim from the context"
    )


class CornersList(BaseModel):
    corners: list[CornerDetails]


class RephraseCandidate(BaseModel):
    rephrase_candidate: str = Field(
        description="one of the candidate rephrases for a given question"
    )


class Rephrase(BaseModel):
    preliminary_questions: list[ReasoningQA]
    is_ambiguous_or_needs_clarification: bool
    is_not_addressable: bool
    final_response: str


class Suggestion(BaseModel):
    quotations: list[str]
    summary: str
    involved_corners: list[str] = Field(
        "how this suggestion can help mitigate the one or more of the given issues in a subset of corners, if provided"
    )


class Counterpoints(BaseModel):
    quotations: list[str] = Field(
        description="reasons, if any, why the suggestion deviates from the user's goal, is detrimental or contrasts previous suggestions"
    )
    summary: str = Field(
        description="summary of counterpoints. If quotations list is empty, then no counterpoint can be raised."
    )


class Recommendation(BaseModel):
    suggestion: Suggestion
    counterpoints: Counterpoints
    verdict: str = Field(
        description="discusses the implications of outputting the current suggestion, and whether the suggestion is worth outputting or it is just noise with respect to the user's goal. Also talks about reasons why the counterpoints are valid and suggestion should be retained."
    )


class Recommendations(BaseModel):
    suggestion_candidates: list[Recommendation]
    final_answer: str = Field(
        description="the actual suggestions for the user. Must consider all previous verdicts, discarding any invalid or dubious, and must be as detailed as possible."
    )
