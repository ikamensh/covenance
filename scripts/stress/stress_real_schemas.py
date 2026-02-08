"""Stress test: real-world complex schemas."""

from pydantic import BaseModel
from stress_utils import DEFAULT_MODEL, StressTestResult, run_test_cases

from covenance import Covenance


# API Response shape
class PaginatedResponse(BaseModel):
    """Typical paginated API response."""

    class PageInfo(BaseModel):
        total: int
        page: int
        per_page: int
        has_next: bool

    class Item(BaseModel):
        id: int
        name: str
        created_at: str
        tags: list[str]

    data: list[Item]
    pagination: PageInfo
    status: str


# Configuration file shape
class AppConfig(BaseModel):
    """Application configuration structure."""

    class DatabaseConfig(BaseModel):
        host: str
        port: int
        name: str
        pool_size: int

    class CacheConfig(BaseModel):
        enabled: bool
        ttl_seconds: int
        max_entries: int

    class LoggingConfig(BaseModel):
        level: str
        format: str
        outputs: list[str]

    app_name: str
    version: str
    debug: bool
    database: DatabaseConfig
    cache: CacheConfig
    logging: LoggingConfig


# Document structure
class Document(BaseModel):
    """Structured document with sections."""

    class Section(BaseModel):
        title: str
        content: str
        subsections: list["Document.Section"] = []

    title: str
    author: str
    abstract: str
    sections: list[Section]
    keywords: list[str]
    references: list[str]


def validate_paginated(result: PaginatedResponse) -> tuple[bool, str]:
    """Validate paginated response structure."""
    issues = []

    if not result.data:
        issues.append("data is empty")
    else:
        if len(result.data) > result.pagination.per_page:
            issues.append(
                f"data has {len(result.data)} items but per_page is {result.pagination.per_page}"
            )
        for i, item in enumerate(result.data):
            if not item.name:
                issues.append(f"item {i} has empty name")
            if item.id <= 0:
                issues.append(f"item {i} has invalid id {item.id}")

    if result.pagination.total < len(result.data):
        issues.append(
            f"total {result.pagination.total} < data length {len(result.data)}"
        )
    if result.pagination.page < 1:
        issues.append(f"page {result.pagination.page} < 1")

    if issues:
        return False, f"Paginated issues: {issues}"
    return True, ""


def validate_config(result: AppConfig) -> tuple[bool, str]:
    """Validate config structure."""
    issues = []

    if not result.app_name:
        issues.append("app_name empty")
    if not result.version:
        issues.append("version empty")

    # Database
    if result.database.port <= 0 or result.database.port > 65535:
        issues.append(f"invalid port {result.database.port}")
    if result.database.pool_size <= 0:
        issues.append(f"invalid pool_size {result.database.pool_size}")

    # Cache
    if result.cache.enabled and result.cache.ttl_seconds <= 0:
        issues.append(f"cache enabled but ttl_seconds {result.cache.ttl_seconds}")

    # Logging
    valid_levels = ["debug", "info", "warning", "error", "critical"]
    if result.logging.level.lower() not in valid_levels:
        issues.append(f"invalid log level {result.logging.level}")
    if not result.logging.outputs:
        issues.append("no logging outputs")

    if issues:
        return False, f"Config issues: {issues}"
    return True, ""


def validate_document(result: Document) -> tuple[bool, str]:
    """Validate document structure."""
    issues = []

    if not result.title:
        issues.append("title empty")
    if not result.author:
        issues.append("author empty")
    if not result.abstract or len(result.abstract) < 20:
        issues.append(
            f"abstract too short: {len(result.abstract) if result.abstract else 0} chars"
        )

    if not result.sections or len(result.sections) < 2:
        issues.append(
            f"need at least 2 sections, got {len(result.sections) if result.sections else 0}"
        )
    else:
        for i, sec in enumerate(result.sections):
            if not sec.title:
                issues.append(f"section {i} has empty title")
            if not sec.content:
                issues.append(f"section {i} has empty content")

    if not result.keywords or len(result.keywords) < 2:
        issues.append("need at least 2 keywords")

    if issues:
        return False, f"Document issues: {issues}"
    return True, ""


def run_stress_test(model: str = DEFAULT_MODEL) -> StressTestResult:
    """Test real-world complex schemas."""
    client = Covenance(label="stress_real")

    cases = [
        (
            "api_response",
            lambda: client.ask_llm(
                "Generate a paginated API response for a list of products. "
                "Include 3 items on page 1 of 5, with 10 total items.",
                model=model,
                response_type=PaginatedResponse,
            ),
            validate_paginated,
        ),
        (
            "app_config",
            lambda: client.ask_llm(
                "Generate a realistic application configuration for a web service "
                "with database, caching, and logging settings.",
                model=model,
                response_type=AppConfig,
            ),
            validate_config,
        ),
        (
            "document",
            lambda: client.ask_llm(
                "Generate a short research paper about machine learning with "
                "title, author, abstract, 3 sections, keywords, and references.",
                model=model,
                response_type=Document,
            ),
            validate_document,
        ),
    ]

    return run_test_cases(client, "real_schemas", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
