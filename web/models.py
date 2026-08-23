from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# Re-exported so API consumers can keep importing the variant vocabulary from
# web.models; the canonical definition lives in web/variants.py.
from web.variants import DEFAULT_VARIANT, PredictionVariant


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
    variant: PredictionVariant
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


class AccountApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class UserPublic(BaseModel):
    id: int
    username: str
    display_name: str
    role: str
    approval_status: AccountApprovalStatus = AccountApprovalStatus.APPROVED


class AccountPublic(UserPublic):
    is_active: bool
    created_at: str
    updated_at: str
    last_login_at: Optional[str] = None
    approved_at: Optional[str] = None
    approved_by_account_id: Optional[int] = None
    rejected_at: Optional[str] = None
    approval_note: str = ""


class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=12, max_length=256)
    admin_code: Optional[str] = None


class SignInRequest(BaseModel):
    username: str
    password: str


class RegistrationResponse(BaseModel):
    status: Literal["approved", "pending_approval"]
    message: str
    account: UserPublic


class ApprovalDecisionRequest(BaseModel):
    note: str = Field(default="", max_length=500)


class AccountCreateRequest(BaseModel):
    username: str = Field(min_length=2, max_length=128)
    display_name: str = Field(min_length=1, max_length=120)
    role: Literal["reviewer", "pi_admin"] = "reviewer"
    password: Optional[str] = Field(default=None, min_length=12, max_length=256)
    approval_note: str = Field(default="", max_length=500)


class AccountCreateResponse(BaseModel):
    account: AccountPublic
    temporary_password: Optional[str] = None


class FeedbackCreate(BaseModel):
    query_id: int
    model_slug: str
    # None means "the deployment's configured default", resolved in the router.
    variant: Optional[PredictionVariant] = None
    outcome: Optional[FeedbackOutcome] = None
    correct_rank: Optional[int] = Field(None, ge=0, le=10)
    correct_dir: Optional[str] = None
    selected_ranks: Optional[List[int]] = None
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
        selected_ranks = values.get("selected_ranks")

        if selected_ranks is not None:
            if not isinstance(selected_ranks, list) or len(selected_ranks) == 0:
                raise ValueError("selected_ranks must be a non-empty list")
            normalized_ranks = []
            for selected_rank in selected_ranks:
                if isinstance(selected_rank, bool):
                    raise ValueError("selected_ranks must contain integers from 1 to 10")
                try:
                    normalized_rank = int(selected_rank)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "selected_ranks must contain integers from 1 to 10"
                    ) from exc
                if not 1 <= normalized_rank <= 10:
                    raise ValueError("selected_ranks must contain integers from 1 to 10")
                normalized_ranks.append(normalized_rank)
            if len(set(normalized_ranks)) != len(normalized_ranks):
                raise ValueError("selected_ranks cannot contain duplicates")
            if outcome not in (None, FeedbackOutcome.MATCHED_RANK.value):
                raise ValueError("selected_ranks can only be used with matched_rank")
            values["selected_ranks"] = normalized_ranks
            outcome = FeedbackOutcome.MATCHED_RANK.value
            values["correct_rank"] = normalized_ranks[0]
            rank = normalized_ranks[0]

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
            values["selected_ranks"] = None
        elif outcome == FeedbackOutcome.SKIPPED.value:
            if rank is not None or correct_dir is not None:
                raise ValueError("skipped cannot include correct_rank or correct_dir")
            values["correct_rank"] = None
            values["correct_dir"] = None
            values["selected_ranks"] = None

        values["outcome"] = outcome
        return values


class FeedbackEntry(BaseModel):
    id: int
    query_id: int
    timestamp: str
    model_slug: str
    # None only for rows written before the variant column existed; those
    # predate the variant CSVs and are never attributed to one retroactively.
    variant: Optional[str] = None
    outcome: FeedbackOutcome
    correct_rank: Optional[int]
    correct_dir: Optional[str]
    selected_ranks: Optional[List[int]] = None
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
    outcome: str
    reviewer: str
    correct_rank: Optional[int] = None


class NeedsAttentionItem(BaseModel):
    file_id: int
    filename: str
    timestamp: str
    model_slug: str
    outcome: str
    notes: str
    reviewer: str


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
    needs_attention: List[NeedsAttentionItem] = []
    next_unreviewed_ids: List[int] = []


# --- Models ---

class ModelInfo(BaseModel):
    slug: str
    display_name: str
    layer: Optional[int] = None
    pooling: Optional[str] = None
    prediction_count: int
    # Variants this deployment can actually serve for this model, plus the one
    # used when a client omits ?variant=.
    available_variants: List[str] = Field(default_factory=list)
    default_variant: str = DEFAULT_VARIANT


# --- Errors ---

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
