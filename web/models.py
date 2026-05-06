from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# --- Queries ---

class QueryListItem(BaseModel):
    file_id: int
    filename: str
    text_preview: str
    review_status: str  # "unreviewed" | "reviewed" | "skipped"
    review_count: int


class QueryListResponse(BaseModel):
    items: List[QueryListItem]
    total: int
    page: int
    page_size: int
    has_more: bool


class NextQueryResponse(BaseModel):
    file_id: Optional[int] = None


class TokenInfo(BaseModel):
    text: str
    index: int
    category: str  # empty / punctuation / number / short_subword / content


class QueryDetail(BaseModel):
    file_id: int
    filename: str
    text: str
    tokens: List[TokenInfo]
    char_count: int
    token_count: int


# --- Predictions ---

class CandidateFile(BaseModel):
    filename: str
    text: str


class Prediction(BaseModel):
    rank: int
    dir_name: str
    score: float
    dir_files: List[str]
    preview_text: str
    candidate_files: Optional[List[CandidateFile]] = None


class PredictionResponse(BaseModel):
    file_id: int
    filename: str
    model: str
    predictions: List[Prediction]


# --- Token Map ---

class TokenEntry(BaseModel):
    idx: int
    text: str
    is_content: bool


class TopMatch(BaseModel):
    candidate_idx: int
    score: float


class AutoHighlight(BaseModel):
    query_idx: int
    ig_score: float
    matches: List[TopMatch]


class TokenMapResponse(BaseModel):
    example_id: int
    model: str
    layer: int
    D: int
    bucket: str
    query_path: str
    candidate_path: str
    query_tokens: List[TokenEntry]
    candidate_tokens: List[TokenEntry]
    similarity_matrix: List[List[float]]
    ig_weighted_matrix: Optional[List[List[float]]] = None
    top_matches: Dict[str, List[TopMatch]]
    query_ig_baseline: List[float]
    query_ig_abtt: List[float]
    candidate_ig_baseline: List[float]
    candidate_ig_abtt: List[float]
    auto_highlights: Optional[List[AutoHighlight]] = None
    available_methods: List[str] = Field(default_factory=list)
    pair_matrices: Dict[str, Dict[str, List[List[float]]]] = Field(default_factory=dict)
    top_highlights: Dict[str, Dict[str, Dict[str, List[int]]]] = Field(default_factory=dict)


class TokenMapExampleSummary(BaseModel):
    example_id: int
    model: str
    bucket: str
    query_path: str
    candidate_path: str


class TokenMapExampleCard(BaseModel):
    example_id: int
    model_slug: str
    bucket: str
    query_file_id: int
    query_folder_id: str
    query_filename: str
    candidate_folder_id: str
    candidate_label: str
    methods_available: List[str]
    gold_similar: int
    baseline_pred: int
    abtt_pred: int


class TokenMapExamplesGroupedResponse(BaseModel):
    by_model: Dict[str, List[TokenMapExampleCard]]
    bucket_order: List[str]


# --- Feedback ---

class FeedbackOutcome(StrEnum):
    MATCHED_RANK = "matched_rank"
    NONE_OF_TOP_K = "none_of_top_k"
    SKIPPED = "skipped"
    LEGACY_UNRESOLVED = "legacy_unresolved"


class UserPublic(BaseModel):
    id: int
    username: str
    display_name: str
    role: str


class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=12, max_length=256)
    admin_code: Optional[str] = None


class SignInRequest(BaseModel):
    username: str
    password: str


class FeedbackCreate(BaseModel):
    query_id: int
    model_slug: str
    outcome: Optional[FeedbackOutcome] = None
    correct_rank: Optional[int] = Field(None, ge=0, le=10)
    correct_dir: Optional[str] = None
    notes: str = ""
    reviewer: str = ""

    @model_validator(mode="before")
    @classmethod
    def validate_outcome_contract(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        values = dict(data)
        explicit_outcome = values.get("outcome") is not None
        outcome = values.get("outcome")
        rank = values.get("correct_rank")
        correct_dir = values.get("correct_dir")

        if outcome is None:
            if rank is None:
                outcome = FeedbackOutcome.LEGACY_UNRESOLVED.value
            elif rank == 0:
                outcome = FeedbackOutcome.NONE_OF_TOP_K.value
            else:
                outcome = FeedbackOutcome.MATCHED_RANK.value

        if outcome == FeedbackOutcome.LEGACY_UNRESOLVED.value and explicit_outcome:
            raise ValueError("legacy_unresolved cannot be created explicitly")

        if outcome == FeedbackOutcome.MATCHED_RANK.value:
            if rank is None or not 1 <= int(rank) <= 10:
                raise ValueError("matched_rank requires correct_rank from 1 to 10")
        elif outcome == FeedbackOutcome.NONE_OF_TOP_K.value:
            if rank is not None and int(rank) != 0:
                raise ValueError("none_of_top_k requires correct_rank 0")
            values["correct_rank"] = 0
            values["correct_dir"] = None
        elif outcome == FeedbackOutcome.SKIPPED.value:
            if rank is not None or correct_dir is not None:
                raise ValueError("skipped cannot include correct_rank or correct_dir")
            values["correct_rank"] = None
            values["correct_dir"] = None

        values["outcome"] = outcome
        return values


class FeedbackEntry(BaseModel):
    id: int
    query_id: int
    timestamp: str
    model_slug: str
    outcome: FeedbackOutcome
    correct_rank: Optional[int]
    correct_dir: Optional[str]
    notes: str
    reviewer: str
    reviewer_account_id: Optional[int] = None
    schema_version: int = 2


# --- Stats ---

class RecentReview(BaseModel):
    file_id: int
    filename: str
    timestamp: str
    model_slug: str


class StatsResponse(BaseModel):
    total_queries: int
    reviewed_count: int
    skipped_count: int = 0
    unreviewed_count: int
    unresolved_count: int = 0
    feedback_count: int
    reviews_by_model: Dict[str, int]
    reviews_by_reviewer: Dict[str, int]
    rank_distribution: Dict[str, int]
    outcome_distribution: Dict[str, int] = {}
    recent_reviews: List[RecentReview] = []
    next_unreviewed_ids: List[int] = []


# --- Models ---

class ModelInfo(BaseModel):
    slug: str
    display_name: str
    layer: Optional[int] = None
    pooling: Optional[str] = None
    prediction_count: int


# --- Errors ---

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
