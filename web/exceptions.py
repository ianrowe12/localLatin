from __future__ import annotations


class QueryNotFoundError(Exception):
    def __init__(self, file_id: int) -> None:
        self.file_id = file_id
        self.message = f"Query file_id={file_id} not found"
        super().__init__(self.message)


class InvalidModelError(Exception):
    def __init__(self, slug: str, available: list[str]) -> None:
        self.slug = slug
        self.available = available
        self.message = f"Unknown model '{slug}'. Available: {available}"
        super().__init__(self.message)


class VariantUnavailableError(Exception):
    """A known variant that this deployment has no predictions CSV for."""

    def __init__(self, variant: str, available: list[str]) -> None:
        self.variant = variant
        self.available = available
        self.message = (
            f"Prediction variant '{variant}' is not available. Available: {available}"
        )
        super().__init__(self.message)


class ExampleNotFoundError(Exception):
    def __init__(self, example_id: int) -> None:
        self.example_id = example_id
        self.message = f"IG example_id={example_id} not found"
        super().__init__(self.message)


class ReviewerDirSeedExistsError(Exception):
    def __init__(self, seed_query_id: int, dir_id: str, label: str) -> None:
        self.seed_query_id = seed_query_id
        self.dir_id = dir_id
        self.message = (
            f"Query {seed_query_id} already seeds reviewer directory "
            f"'{dir_id}' ({label})."
        )
        super().__init__(self.message)


class ReviewerDirLimitError(Exception):
    def __init__(self, created_count: int, limit: int) -> None:
        self.created_count = created_count
        self.limit = limit
        self.message = (
            f"You have created {created_count} reviewer directories, the "
            f"maximum is {limit}. Ask a PI/admin if you genuinely need more."
        )
        super().__init__(self.message)
