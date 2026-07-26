"""SEC Financial Entity Extractor for GraphRAG."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

COMPANY_MAPPING: dict[str, str] = {
    "aapl": "Apple Inc.",
    "apple": "Apple Inc.",
    "apple inc": "Apple Inc.",
    "xom": "ExxonMobil",
    "exxon": "ExxonMobil",
    "exxonmobil": "ExxonMobil",
    "exxon mobil": "ExxonMobil",
    "msft": "Microsoft",
    "microsoft": "Microsoft",
    "nvda": "Nvidia",
    "nvidia": "Nvidia",
    "amzn": "Amazon",
    "amazon": "Amazon",
    "amazon.com": "Amazon",
    "jpm": "JPMorgan Chase",
    "chase": "JPMorgan Chase",
    "jpmorgan": "JPMorgan Chase",
    "jpmorgan chase": "JPMorgan Chase",
    "jp morgan": "JPMorgan Chase",
    "nflx": "Netflix, Inc.",
    "netflix": "Netflix, Inc.",
    "netflix inc": "Netflix, Inc.",
    "bkng": "Booking Holdings",
    "booking": "Booking Holdings",
    "goog": "Alphabet Inc. (Class A)",
    "googl": "Alphabet Inc. (Class A)",
    "google": "Alphabet Inc. (Class A)",
    "alphabet": "Alphabet Inc. (Class A)",
    "meta": "Meta Platforms",
    "facebook": "Meta Platforms",
    "tsla": "Tesla, Inc.",
    "tesla": "Tesla, Inc.",
    "dis": "Walt Disney Company (The)",
    "disney": "Walt Disney Company (The)",
    "wmt": "Walmart",
    "walmart": "Walmart",
    "pg": "Procter & Gamble",
    "procter": "Procter & Gamble",
    "procter & gamble": "Procter & Gamble",
    "v": "Visa Inc.",
    "visa": "Visa Inc.",
    "ma": "Mastercard",
    "mastercard": "Mastercard",
    "jnj": "Johnson & Johnson",
    "bac": "Bank of America",
    "ko": "Coca-Cola Company (The)",
    "coca-cola": "Coca-Cola Company (The)",
    "coca cola": "Coca-Cola Company (The)",
    "coke": "Coca-Cola Company (The)",
    "csco": "Cisco",
    "cisco": "Cisco",
    "cat": "Caterpillar Inc.",
    "caterpillar": "Caterpillar Inc.",
    "cvx": "Chevron Corporation",
    "chevron": "Chevron Corporation",
    "abbv": "AbbVie",
    "abbvie": "AbbVie",
    "adbe": "Adobe Inc.",
    "adobe": "Adobe Inc.",
    "amd": "Advanced Micro Devices",
    "advanced micro devices": "Advanced Micro Devices",
    "intc": "Intel",
    "intel": "Intel",
}

KNOWN_EXECUTIVES: list[str] = [
    "John Ternus",
    "Tim Cook",
    "Andy Jassy",
    "Satya Nadella",
    "Jensen Huang",
    "Sundar Pichai",
    "Mark Zuckerberg",
    "Elon Musk",
    "Bob Iger",
    "Doug McMillon",
    "Jamie Dimon",
    "Darren Woods",
    "Warren Buffett",
    "Ted Sarandos",
    "Greg Peters",
    "Jeff Bezos",
]

AUDITORS_LIST: list[str] = [
    "PricewaterhouseCoopers",
    "PwC",
    "Ernst & Young",
    "EY",
    "Deloitte",
    "KPMG",
]

EVENTS_LIST: list[str] = [
    "CEO Transition",
    "appointment",
    "appointed",
    "merger",
    "acquisition",
    "restructuring",
    "dividend",
    "stock split",
    "buyback",
    "lawsuit",
]

METRICS_LIST: list[str] = [
    "total revenues and other income",
    "total net sales",
    "net sales",
    "total revenue",
    "total revenues",
    "revenue",
    "net income",
    "operating income",
    "gross profit",
    "earnings per share",
    "eps",
    "operating cash flow",
    "free cash flow",
    "total assets",
    "total liabilities",
    "operating expenses",
]

RISKS_LIST: list[str] = [
    "cybersecurity",
    "regulatory",
    "competition",
    "supply chain",
    "inflation",
    "foreign exchange",
    "market risk",
    "cloud provider",
    "streaming service",
    "aws",
    "infrastructure",
]

FILING_TYPES: list[str] = [
    "10-K",
    "10-Q",
    "8-K",
    "DEF14A",
    "DEF 14A",
    "annual report",
    "proxy statement",
]


@dataclass
class ExtractedEntities:
    companies: list[str] = field(default_factory=list)
    executives: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    auditors: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    filing_types: list[str] = field(default_factory=list)
    years: list[str] = field(default_factory=list)

    @property
    def priority_ordered_list(self) -> list[str]:
        """Search using entity priority:
        1. Company
        2. Executive
        3. Ticker
        4. Auditor
        5. Event
        6. Risk
        7. Financial Metric
        8. Filing Type
        9. Fiscal Year
        """
        ordered = []
        for item in (
            self.companies
            + self.executives
            + self.tickers
            + self.auditors
            + self.events
            + self.risks
            + self.metrics
            + self.filing_types
            + self.years
        ):
            if item not in ordered:
                ordered.append(item)
        return ordered


def extract_entities(
    query: str, known_entities: list[tuple[str, str]] | None = None
) -> ExtractedEntities:
    """Deterministic rule-based Python entity extractor for GraphRAG."""
    query_lower = query.lower()
    cleaned_query = re.sub(r"['’]s\b", "", query_lower)

    extracted = ExtractedEntities()

    # 1. Companies & Tickers
    seen_companies = set()
    for kw in sorted(COMPANY_MAPPING.keys(), key=len, reverse=True):
        pattern = r"\b" + re.escape(kw) + r"(?:['’]?s)?\b"
        if re.search(pattern, query_lower) or re.search(pattern, cleaned_query):
            canonical = COMPANY_MAPPING[kw]
            if canonical not in seen_companies:
                extracted.companies.append(canonical)
                seen_companies.add(canonical)
                if kw.isupper() and len(kw) <= 5:
                    extracted.tickers.append(kw.upper())

    if known_entities:
        for name, vid in known_entities:
            if name in seen_companies:
                continue
            name_low = name.lower()
            if re.search(r"\b" + re.escape(name_low) + r"\b", query_lower) or re.search(
                r"\b" + re.escape(name_low) + r"\b", cleaned_query
            ):
                extracted.companies.append(name)
                seen_companies.add(name)

    # 2. Executives
    for ex in KNOWN_EXECUTIVES:
        if ex.lower() in query_lower and ex not in extracted.executives:
            extracted.executives.append(ex)

    if not extracted.executives:
        match = re.search(
            r"\b(?:appointed|hired|named|appointed\s+as\s+CEO|as\s+CEO|CEO|Chief Executive Officer)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
            query,
        )
        if match:
            extracted.executives.append(match.group(1))

    # 3. Auditors
    for aud in AUDITORS_LIST:
        if aud.lower() in query_lower and aud not in extracted.auditors:
            extracted.auditors.append(aud)

    # 4. Events
    for ev in EVENTS_LIST:
        if ev.lower() in query_lower and ev not in extracted.events:
            extracted.events.append(ev.title())

    # 5. Risks & Keywords
    for rk in RISKS_LIST:
        if rk.lower() in query_lower and rk not in extracted.risks:
            extracted.risks.append(rk.title())

    # 6. Financial Metrics
    seen_metrics = set()
    for m in sorted(METRICS_LIST, key=len, reverse=True):
        if m in query_lower:
            if not any(m in existing.lower() for existing in seen_metrics):
                title_m = m.title()
                extracted.metrics.append(title_m)
                seen_metrics.add(m)

    # 7. Filing Types
    for ft in FILING_TYPES:
        if ft.lower() in query_lower and ft not in extracted.filing_types:
            extracted.filing_types.append(ft)

    # 8. Fiscal Years
    years = re.findall(r"\b(?:fy\s*)?(20\d{2})\b", query, re.IGNORECASE)
    for y in years:
        if y not in extracted.years:
            extracted.years.append(y)

    return extracted


def determine_question_type(extracted: ExtractedEntities) -> str:
    """Determine query routing type based on extracted entities:
    - Comparison: 2+ companies
    - Executive: Executive detected
    - Auditor: Auditor detected
    - Event: Event detected
    - Company: 1 company detected
    - Fallback: No company/executive identified
    """
    if len(extracted.companies) >= 2:
        return "comparison"
    if extracted.executives:
        return "executive"
    if extracted.auditors:
        return "auditor"
    if extracted.events:
        return "event"
    if extracted.companies:
        return "company"
    return "fallback"
