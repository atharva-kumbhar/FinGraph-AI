"""BERTScore and LLM-as-judge evaluation hooks."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .config import settings
from .llm import LLMClient


@dataclass(frozen=True)
class AccuracyResult:
    judge: str
    pass_rate: float | None
    bertscore_f1: float | None
    hallucination_risk: str
    method: str
    rationale: str


class AccuracyEvaluator:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def score(
        self,
        query: str,
        answer: str,
        evidence: str,
        pipeline: str,
        reference_answer: str | None = None,
    ) -> AccuracyResult:
        if not answer:
            return AccuracyResult(
                judge="NOT_EVALUATED",
                pass_rate=None,
                bertscore_f1=None,
                hallucination_risk="UNKNOWN",
                method="no answer generated",
                rationale="The pipeline did not produce an answer.",
            )

        bertscore_f1 = self._bertscore(answer, reference_answer) if reference_answer else None
        if not reference_answer:
            return AccuracyResult(
                judge="NOT_EVALUATED",
                pass_rate=None,
                bertscore_f1=bertscore_f1,
                hallucination_risk="UNKNOWN",
                method="requires expected answer",
                rationale="Generate or provide expected_answer/reference_answer to compute LLM judge accuracy and BERTScore.",
            )
        if not settings.enable_llm_judge:
            return AccuracyResult(
                judge="NOT_EVALUATED",
                pass_rate=None,
                bertscore_f1=bertscore_f1,
                hallucination_risk="UNKNOWN",
                method="LLM-as-judge disabled",
                rationale="Set MEDGRAPH_ENABLE_LLM_JUDGE=true to evaluate generated answers.",
            )
        if not self.llm_client or not self.llm_client.has_provider("gemini"):
            return AccuracyResult(
                judge="NOT_EVALUATED",
                pass_rate=None,
                bertscore_f1=bertscore_f1,
                hallucination_risk="UNKNOWN",
                method="requires hosted LLM judge",
                rationale="Set GEMINI_API_KEY for Gemini LLM-as-a-judge scoring.",
            )

        prompt = self._judge_prompt(
            query=query,
            answer=answer,
            evidence=evidence,
            pipeline=pipeline,
            reference_answer=reference_answer,
            bertscore_f1=bertscore_f1,
        )
        try:
            generation = self.llm_client.generate(
                prompt,
                system=(
                    "You are a strict SEC filing and corporate finance QA evaluator. Rate the generated answer from 1 to 10 "
                    "for factual correctness, financial data accuracy, and relevance compared to the expected financial answer. "
                    "Ensure your output format ends with 'Score: [number]'."
                ),
                provider="gemini",
                temperature=0.0,
                max_tokens=150,
            )
        except Exception as exc:
            fallback = self._fallback_judge(bertscore_f1)
            score_val = 10.0 if fallback == "PASS" else 2.0
            return AccuracyResult(
                judge=str(score_val),
                pass_rate=score_val * 10.0,
                bertscore_f1=bertscore_f1,
                hallucination_risk="UNKNOWN",
                method="LLM-as-judge failed",
                rationale=self._friendly_external_error(exc),
            )

        # Parse 1-10 score
        score_match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", generation.answer, re.IGNORECASE)
        if not score_match:
            score_match = re.search(r"\b([1-9]|10)\b", generation.answer)
        
        score_val = None
        if score_match:
            try:
                score_val = float(score_match.group(1))
                score_val = max(1.0, min(10.0, score_val))
            except ValueError:
                pass
        
        if score_val is None:
            fallback = self._fallback_judge(bertscore_f1)
            score_val = 10.0 if fallback == "PASS" else 2.0

        accuracy = score_val * 10.0
        return AccuracyResult(
            judge=str(score_val),
            pass_rate=accuracy,
            bertscore_f1=bertscore_f1,
            hallucination_risk="UNKNOWN",
            method="LLM-as-a-judge (1-10)",
            rationale=generation.answer.strip(),
        )

    @dataclass(frozen=True)
    class TempResult:
        pass

    @staticmethod
    def _judge_prompt(
        *,
        query: str,
        answer: str,
        evidence: str,
        pipeline: str,
        reference_answer: str | None,
        bertscore_f1: float | None,
    ) -> str:
        reference_block = reference_answer or ""
        evidence_block = evidence or "No retrieved evidence was provided by this pipeline."
        return f"""
Evaluate the generated answer for an SEC financial QA system.

Pipeline: {pipeline}
User query:
{query}

Expected financial answer:
{reference_block}

Generated answer to evaluate:
{answer}

Retrieved or graph evidence:
{evidence_block}

Instructions:
1. Compare the Expected Financial Answer vs the Generated Answer.
2. Score the Generated Answer from 1 to 10 based on factual correctness, accuracy, and relevance.
3. Provide a brief rationale.
4. Your response must end with exactly: "Score: [value]" (e.g. "Score: 8").
"""

    @staticmethod
    def _bertscore(answer: str, reference_answer: str | None) -> float | None:
        if not reference_answer:
            return None
        if not settings.enable_bertscore:
            return AccuracyEvaluator._semantic_f1(answer, reference_answer)
        try:
            from bert_score import score as bert_score
        except ImportError:
            return AccuracyEvaluator._semantic_f1(answer, reference_answer)
        _, _, f1 = bert_score([answer], [reference_answer], lang="en", verbose=False)
        return round(float(f1.mean().item()), 4)

    @staticmethod
    def _semantic_f1(answer: str, reference_answer: str | None) -> float | None:
        if not reference_answer:
            return None
        answer_terms = AccuracyEvaluator._content_terms(answer)
        reference_terms = AccuracyEvaluator._content_terms(reference_answer)
        if not answer_terms or not reference_terms:
            return None
        overlap = answer_terms & reference_terms
        precision = len(overlap) / len(answer_terms)
        recall = len(overlap) / len(reference_terms)
        if math.isclose(precision + recall, 0.0):
            return 0.0
        return round((2 * precision * recall) / (precision + recall), 4)

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        stopwords = {
            "about",
            "after",
            "and",
            "are",
            "because",
            "before",
            "consider",
            "for",
            "from",
            "has",
            "have",
            "into",
            "not",
            "patient",
            "patients",
            "should",
            "that",
            "the",
            "their",
            "this",
            "with",
            "would",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", (text or "").lower())
            if len(token) >= 3 and token not in stopwords
        }

    @staticmethod
    def _parse_pass_fail(text: str) -> str | None:
        normalized = (text or "").strip().upper()
        if re.search(r"\bFAIL\b", normalized):
            return "FAIL"
        if re.search(r"\bPASS\b", normalized):
            return "PASS"
        return None

    @staticmethod
    def _fallback_judge(bertscore_f1: float | None) -> str | None:
        if bertscore_f1 is None:
            return None
        return "PASS" if bertscore_f1 >= 0.42 else "FAIL"

    @staticmethod
    def _friendly_external_error(exc: Exception) -> str:
        message = str(exc)
        if "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
            return "Gemini quota/rate limit reached; used semantic fallback where possible."
        return "Hosted judge request failed; used semantic fallback where possible."
