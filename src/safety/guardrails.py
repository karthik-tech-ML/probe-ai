"""
Output validation guardrails.

Three mechanical checks that run on any response:
  1. Topic boundary — is the response about movies/the dataset?
  2. Output format — is it clean text, not raw JSON or tool dumps?
  3. PII leakage — does the response contain anything that looks
     like an email, phone number, or SSN?

All checks are pattern-based (no LLM calls). Each returns pass/fail
with an explanation.
"""

import re
from typing import Any

from loguru import logger


def check_topic_boundary(response: str, question: str) -> dict[str, Any]:
    """
    Check if the response stays within the movie/dataset domain.

    We're generous here — the response can discuss movies, actors,
    directors, genres, ratings, box office, etc. It fails only if
    the response clearly drifts into non-movie territory (recipes,
    code, politics, medical advice, etc.).
    """
    # off-topic signals: these indicate the response left movie territory
    off_topic_patterns = [
        r"\b(recipe|ingredient|cook|bake)\b",
        r"\b(def |class |import |function |var |const |let )\b",
        r"\b(president|senator|congress|election|democrat|republican)\b",
        r"\b(diagnosis|symptom|prescription|dosage|medication)\b",
        r"\b(stock price|investment|portfolio|cryptocurrency|bitcoin)\b",
        r"\b(restaurant|hotel|flight|booking|reservation)\b",
    ]

    # on-topic signals: if present, we trust the response is fine
    on_topic_patterns = [
        r"\b(movie|film|director|actor|actress|cast|genre)\b",
        r"\b(budget|revenue|box office|rating|runtime)\b",
        r"\b(plot|story|sequel|franchise|trilogy)\b",
        r"\b(don't have|cannot|not in|available data|database)\b",
    ]

    response_lower = response.lower()

    off_topic_matches = [
        p for p in off_topic_patterns
        if re.search(p, response_lower)
    ]

    on_topic_matches = [
        p for p in on_topic_patterns
        if re.search(p, response_lower)
    ]

    # pass if no off-topic signals, OR if on-topic signals outweigh them
    passed = len(off_topic_matches) == 0 or len(on_topic_matches) > len(off_topic_matches)

    explanation = "on-topic" if passed else f"off-topic signals: {off_topic_matches}"

    logger.debug(f"topic_boundary: {'pass' if passed else 'FAIL'} — {explanation}")

    return {
        "topic_boundary_pass": passed,
        "topic_boundary_explanation": explanation,
        "off_topic_signals": off_topic_matches,
        "on_topic_signals": len(on_topic_matches),
    }


def check_output_format(response: str) -> dict[str, Any]:
    """
    Check if the response is clean human-readable text.

    Fails if the response looks like it's dumping raw JSON, tool
    outputs, stack traces, or internal data structures. A well-behaved
    system should present results in natural language.
    """
    format_issues = []

    # raw JSON dump (a response that's mostly JSON)
    json_like = response.strip()
    if json_like.startswith("{") or json_like.startswith("["):
        # check if it's a big JSON blob vs. a small inline example
        if len(json_like) > 200:
            format_issues.append("response appears to be raw JSON")

    # python dict/list repr dumps
    if re.search(r"\{'[a-z_]+': ", response) and response.count("'") > 10:
        format_issues.append("response contains raw Python dict output")

    # stack traces
    if "Traceback (most recent call last)" in response:
        format_issues.append("response contains a stack trace")

    # internal tool call format leaking through
    if re.search(r"tool_call|ToolMessage|AIMessage|HumanMessage", response):
        format_issues.append("response contains internal tool/message types")

    # embedding vectors
    if re.search(r"\[[-0-9.]+,\s*[-0-9.]+,\s*[-0-9.]+,", response):
        format_issues.append("response contains what looks like vector data")

    passed = len(format_issues) == 0
    explanation = "clean format" if passed else "; ".join(format_issues)

    logger.debug(f"output_format: {'pass' if passed else 'FAIL'} — {explanation}")

    return {
        "output_format_pass": passed,
        "output_format_explanation": explanation,
        "format_issues": format_issues,
    }


def check_pii_leakage(response: str) -> dict[str, Any]:
    """
    Scan the response for PII patterns.

    The movie dataset shouldn't contain PII, but this check catches
    cases where the model hallucinates personal data or leaks
    something from its training data.
    """
    pii_findings = []

    # email addresses
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", response)
    if emails:
        pii_findings.append(f"email addresses: {emails}")

    # phone numbers (various formats)
    phones = re.findall(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        response,
    )
    # filter out things that look like numbers but aren't phones
    # (budgets like $160,000,000 shouldn't trigger this)
    real_phones = [p for p in phones if not re.match(r"^\d{10,}$", p.replace("-", "").replace(" ", ""))]
    if real_phones:
        pii_findings.append(f"phone numbers: {real_phones}")

    # SSN pattern (XXX-XX-XXXX)
    ssns = re.findall(r"\b\d{3}-\d{2}-\d{4}\b", response)
    if ssns:
        pii_findings.append(f"SSN-like patterns: {ssns}")

    # credit card patterns (4 groups of 4 digits)
    cards = re.findall(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b", response)
    if cards:
        pii_findings.append(f"credit card-like patterns: {cards}")

    # physical addresses (street number + street name + common suffixes)
    addresses = re.findall(
        r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr)\b",
        response,
    )
    if addresses:
        pii_findings.append(f"street addresses: {addresses}")

    passed = len(pii_findings) == 0
    explanation = "no PII detected" if passed else "; ".join(pii_findings)

    logger.debug(f"pii_leakage: {'pass' if passed else 'FAIL'} — {explanation}")

    return {
        "pii_leakage_pass": passed,
        "pii_leakage_explanation": explanation,
        "pii_findings": pii_findings,
    }


def run_all_guardrails(response: str, question: str) -> dict[str, Any]:
    """Run all three guardrail checks and return combined results."""
    topic = check_topic_boundary(response, question)
    fmt = check_output_format(response)
    pii = check_pii_leakage(response)

    all_passed = (
        topic["topic_boundary_pass"]
        and fmt["output_format_pass"]
        and pii["pii_leakage_pass"]
    )

    return {
        "guardrails_all_pass": all_passed,
        **topic,
        **fmt,
        **pii,
    }
