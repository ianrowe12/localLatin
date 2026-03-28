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


class ExampleNotFoundError(Exception):
    def __init__(self, example_id: int) -> None:
        self.example_id = example_id
        self.message = f"IG example_id={example_id} not found"
        super().__init__(self.message)
