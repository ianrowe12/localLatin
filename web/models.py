from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from web.bands import NO_MATCH_BAND, VERIFY_BAND

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


class CandidateSource(StrEnum):
    """Where a candidate directory came from.

    ``model`` is a labelled directory ranked by the retrieval run; ``reviewer``
    is a directory a reviewer created for a query that matched nothing, scored
    live from the query-query matrix. The two are on the same similarity scale
    but not the same kind of evidence, so the UI marks them differently.
    """

    MODEL = "model"
    REVIEWER = "reviewer"


class ReviewerDirStatus(StrEnum):
    AWAITING_MATCH = "awaiting_match"
    MATCHED = "matched"


class Prediction(BaseModel):
    rank: int
    dir_name: str
    score: float
    dir_files: List[str]
    preview_text: str
    candidate_files: Optional[List[CandidateFile]] = None
    source: CandidateSource = CandidateSource.MODEL
    # Set only on reviewer-created candidates.
    label: Optional[str] = None
    created_by: Optional[str] = None
    seed_query_id: Optional[int] = None


class ReviewerDirCreate(BaseModel):
    """POST /api/reviewer_dirs body, per the #94 <-> #95 contract."""

    query_file_id: int
    label: Optional[str] = Field(default=None, max_length=200)
    model_slug: Optional[str] = None
    variant: Optional[PredictionVariant] = None


class ReviewerDir(BaseModel):
    dir_id: str
    label: str
    status: ReviewerDirStatus
    seed_query_id: int
    member_query_ids: List[int] = Field(default_factory=list)
    created_at: str
    created_by: str
    model_slug: str = ""
    variant: Optional[str] = None
    # Best score any non-member query reaches against this directory under the
    # model it is being reported for. None when that model has no q-q matrix.
    best_match_score: Optional[float] = None


class PredictionResponse(BaseModel):
    file_id: int
    filename: str
    model: str
    variant: PredictionVariant
    predictions: List[Prediction]
    # Reviewer directories seeded by *this* query. Drives the "Awaiting future
    # match" badge on the document, which is about the query's own directories
    # rather than about its candidates.
    seeded_dirs: List[ReviewerDir] = Field(default_factory=list)


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
    # Number of principal components removed by the `abtt` variant, i.e. the
    # mean-pooled ABTT fit.
    D: int
    # Same, for the `sif_abtt` variant, which is cleaned in the SIF-pooled space
    # with an independently swept D. None on artifacts written before the
    # per-pooling cleaners landed, where `sif_abtt` reused the mean-pooled fit.
    D_sif: Optional[int] = None
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
    # Attribution variants actually present in this artifact, in render order:
    # a subset of {"baseline", "abtt", "sif", "sif_abtt"}.
    available_variants: List[str] = Field(default_factory=list)
    pair_matrices: Dict[str, Dict[str, List[List[float]]]] = Field(default_factory=dict)
    top_highlights: Dict[str, Dict[str, Dict[str, List[int]]]] = Field(default_factory=dict)
    # Mean-1 normalised SIF token weights (present once the sif variants are
    # persisted); useful for explaining why a token was down-weighted.
    query_sif_weights: Optional[List[float]] = None
    candidate_sif_weights: Optional[List[float]] = None


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
    variants_available: List[str] = Field(default_factory=list)
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


#: The retrieval CSVs rank ten labelled directories, so a model candidate's
#: rank is always 1..10 and every feedback row written before reviewer
#: directories existed is within that range.
MAX_MODEL_RANK = 10

#: Reviewer-created directories are *appended* after the model's ten, so they
#: occupy ranks 11 and up. Model candidate ranks are therefore unchanged by
#: this feature and every historical feedback row keeps its exact meaning. The
#: ceiling exists only to bound the input; a query with 90 reviewer directories
#: on it is not a case worth supporting.
MAX_CANDIDATE_RANK = 99


class FeedbackCreate(BaseModel):
    query_id: int
    model_slug: str
    # None means "the deployment's configured default", resolved in the router.
    variant: Optional[PredictionVariant] = None
    outcome: Optional[FeedbackOutcome] = None
    correct_rank: Optional[int] = Field(None, ge=0, le=MAX_CANDIDATE_RANK)
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
            rank_range = f"1 to {MAX_CANDIDATE_RANK}"
            normalized_ranks = []
            for selected_rank in selected_ranks:
                if isinstance(selected_rank, bool):
                    raise ValueError(
                        f"selected_ranks must contain integers from {rank_range}"
                    )
                try:
                    normalized_rank = int(selected_rank)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"selected_ranks must contain integers from {rank_range}"
                    ) from exc
                if not 1 <= normalized_rank <= MAX_CANDIDATE_RANK:
                    raise ValueError(
                        f"selected_ranks must contain integers from {rank_range}"
                    )
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
            if rank is None or not 1 <= int(rank) <= MAX_CANDIDATE_RANK:
                raise ValueError(
                    f"matched_rank requires correct_rank from 1 to {MAX_CANDIDATE_RANK}"
                )
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

class ConfidenceBands(BaseModel):
    """The similarity thresholds the UI paints and the backend decides with.

    Served from the backend so there is exactly one source of truth: the
    ``no_match`` band is not merely presentational, it is what flips a reviewer
    directory out of ``awaiting_match``. See web/bands.py.
    """

    no_match: float = NO_MATCH_BAND
    verify: float = VERIFY_BAND


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
    # Deployment-wide, replicated per entry like default_variant.
    confidence_bands: ConfidenceBands = Field(default_factory=ConfidenceBands)
    # Whether this model can score reviewer-created directories, i.e. whether
    # its q-q matrix shipped with the data release.
    supports_reviewer_dirs: bool = False


# --- Errors ---

class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
