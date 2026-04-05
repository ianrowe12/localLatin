from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --- Queries ---

class QueryListItem(BaseModel):
    file_id: int
    filename: str
    text_preview: str
    review_status: str  # "unreviewed" | "reviewed"
    review_count: int


class QueryListResponse(BaseModel):
    items: List[QueryListItem]
    total: int
    page: int
    page_size: int
    has_more: bool


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


class TokenMapExampleSummary(BaseModel):
    example_id: int
    model: str
    bucket: str
    query_path: str
    candidate_path: str


# --- Feedback ---

class FeedbackCreate(BaseModel):
    query_id: int
    model_slug: str
    correct_rank: Optional[int] = Field(None, ge=0, le=10)
    correct_dir: Optional[str] = None
    notes: str = ""
    reviewer: str


class FeedbackEntry(BaseModel):
    id: int
    query_id: int
    timestamp: str
    model_slug: str
    correct_rank: Optional[int]
    correct_dir: Optional[str]
    notes: str
    reviewer: str


# --- Stats ---

class RecentReview(BaseModel):
    file_id: int
    filename: str
    timestamp: str
    model_slug: str


class StatsResponse(BaseModel):
    total_queries: int
    reviewed_count: int
    unreviewed_count: int
    feedback_count: int
    reviews_by_model: Dict[str, int]
    reviews_by_reviewer: Dict[str, int]
    rank_distribution: Dict[str, int]
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
