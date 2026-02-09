"""Stress test: string format adherence.

Tests whether LLMs follow format instructions in field descriptions
for common patterns: email, URL, ISO date, UUID, etc.
"""

import re

from pydantic import BaseModel, Field
from stress_utils import DEFAULT_MODEL, StressTestResult, make_client, run_test_cases

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
URL_RE = re.compile(r"^https?://[^\s]+$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class ContactInfo(BaseModel):
    name: str = Field(description="A person's full name")
    email: str = Field(description="A valid email address like user@example.com")
    website: str = Field(description="A valid URL starting with https://")
    phone: str = Field(description="Phone number in format +1-555-123-4567")


class TimestampRecord(BaseModel):
    event: str
    date: str = Field(description="Date in ISO 8601 format: YYYY-MM-DD")
    datetime_val: str = Field(
        description="Datetime in ISO 8601 format: YYYY-MM-DDTHH:MM:SS"
    )


class IdentifierRecord(BaseModel):
    id: str = Field(
        description="A UUID v4 in standard format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    )
    version: str = Field(description="A semantic version like 1.2.3")
    color: str = Field(description="A hex color code like #FF5733")


class MixedFormats(BaseModel):
    email: str = Field(description="Valid email address")
    date: str = Field(description="ISO date YYYY-MM-DD")
    url: str = Field(description="Valid HTTPS URL")
    uuid: str = Field(description="UUID v4")


def validate_contact(result: ContactInfo) -> tuple[bool, str]:
    issues = []
    if not result.name or len(result.name) < 2:
        issues.append(f"name too short: '{result.name}'")
    if not EMAIL_RE.match(result.email):
        issues.append(f"bad email: '{result.email}'")
    if not URL_RE.match(result.website):
        issues.append(f"bad URL: '{result.website}'")
    if not result.phone or len(result.phone) < 8:
        issues.append(f"bad phone: '{result.phone}'")
    if issues:
        return False, f"Contact issues: {issues}"
    return True, ""


def validate_timestamps(result: TimestampRecord) -> tuple[bool, str]:
    issues = []
    if not result.event:
        issues.append("event empty")
    if not ISO_DATE_RE.match(result.date):
        issues.append(f"bad date: '{result.date}'")
    if not ISO_DATETIME_RE.match(result.datetime_val):
        issues.append(f"bad datetime: '{result.datetime_val}'")
    if issues:
        return False, f"Timestamp issues: {issues}"
    return True, ""


def validate_identifiers(result: IdentifierRecord) -> tuple[bool, str]:
    issues = []
    if not UUID_RE.match(result.id):
        issues.append(f"bad UUID: '{result.id}'")
    if not SEMVER_RE.match(result.version):
        issues.append(f"bad semver: '{result.version}'")
    if not HEX_COLOR_RE.match(result.color):
        issues.append(f"bad hex color: '{result.color}'")
    if issues:
        return False, f"Identifier issues: {issues}"
    return True, ""


def validate_mixed(result: MixedFormats) -> tuple[bool, str]:
    issues = []
    if not EMAIL_RE.match(result.email):
        issues.append(f"bad email: '{result.email}'")
    if not ISO_DATE_RE.match(result.date):
        issues.append(f"bad date: '{result.date}'")
    if not URL_RE.match(result.url):
        issues.append(f"bad URL: '{result.url}'")
    if not UUID_RE.match(result.uuid):
        issues.append(f"bad UUID: '{result.uuid}'")
    if issues:
        return False, f"Mixed format issues: {issues}"
    return True, ""


def run_stress_test(
    model: str = DEFAULT_MODEL, backend: str | None = None
) -> StressTestResult:
    """Test string format adherence."""
    client = make_client(model, backend)

    cases = [
        (
            "contact_formats",
            lambda: client.ask_llm(
                "Generate a fictional contact with valid email, website URL, and phone number.",
                model=model,
                response_type=ContactInfo,
            ),
            validate_contact,
        ),
        (
            "timestamp_formats",
            lambda: client.ask_llm(
                "Create an event record for a meeting on 2024-03-15 at 14:30:00.",
                model=model,
                response_type=TimestampRecord,
            ),
            validate_timestamps,
        ),
        (
            "identifier_formats",
            lambda: client.ask_llm(
                "Generate a record with a UUID, semantic version, and hex color.",
                model=model,
                response_type=IdentifierRecord,
            ),
            validate_identifiers,
        ),
        (
            "all_formats_combined",
            lambda: client.ask_llm(
                "Generate a record with a valid email, ISO date, HTTPS URL, and UUID.",
                model=model,
                response_type=MixedFormats,
            ),
            validate_mixed,
        ),
    ]

    return run_test_cases(client, "formats", cases)


if __name__ == "__main__":
    result = run_stress_test()
    print(f"{'PASS' if result.passed else 'FAIL'}: {result.name}")
    for f in result.failures:
        print(f"  - {f}")
