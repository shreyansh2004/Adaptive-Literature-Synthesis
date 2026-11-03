"""
Fast, no-Ollama smoke tests for resume/CI signals.

Run: pytest tests/test_smoke.py -q
"""

import re

from agents.planner import (
    query_is_self_referential,
    query_looks_gibberish,
    query_requests_comparison,
)
from agents.synthesizer import (
    clamp_citation_indices,
    fix_literal_n_citations,
    _query_wants_multi_paper,
    _rank_grouped_by_title,
)
from collections import OrderedDict
from rag.retrieve import (
    aspect_keywords,
    split_comparison_aspects,
    _aspect_title_score,
    _is_definitional_query,
    _wants_paper_diversity,
)


def test_differ_inside_different_does_not_force_compare():
    q = "What does layer-wise LoRA adaptation mean, and why might different layers need different treatment?"
    assert not query_requests_comparison(q)
    assert not _query_wants_multi_paper(q)


def test_difference_between_does_force_compare():
    q = "What's the difference between LoRA fine-tuning and partial fine-tuning approaches?"
    assert query_requests_comparison(q)
    assert _query_wants_multi_paper(q)


def test_gibberish_and_self_ref_gates():
    assert query_looks_gibberish("asdkjfh laksjdf compare papers???")
    assert query_is_self_referential("What embedding model does this system use?")


def test_clamp_and_literal_n_citations():
    clamped = clamp_citation_indices("LoRA works well [1] and also [99].", 2)
    assert "[1]" in clamped
    assert "99" not in clamped
    assert fix_literal_n_citations("Answer [n] here.", 1) == "Answer [1] here."
    assert "[n]" not in fix_literal_n_citations("Maybe [n] or [n].", 2).lower()


def test_aspect_split_and_partial_keywords():
    q = "What's the difference between LoRA fine-tuning and partial fine-tuning approaches?"
    aspects = split_comparison_aspects(q)
    assert len(aspects) == 2
    assert "partial" in aspect_keywords(aspects[1])
    assert "fine" in aspect_keywords(aspects[1])


def test_canonical_lora_beats_variant_on_standard_aspect():
    aspect = "standard LoRA"
    canonical = _aspect_title_score(
        "LoRA: Low-Rank Adaptation of Large Language Models", aspect
    )
    variant = _aspect_title_score(
        "LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation", aspect
    )
    assert canonical > variant


def test_definitional_and_diversity_flags():
    assert _is_definitional_query("What is retrieval-augmented generation (RAG)?")
    assert _wants_paper_diversity(
        "Do different papers agree on how LoRA performance compares to full fine-tuning?"
    )
    assert not _wants_paper_diversity("What is LoRA fine-tuning?")


def test_title_rank_prefers_leading_query_terms():
    q = "How does federated learning combine with LoRA fine-tuning?"
    grouped = OrderedDict(
        [
            (
                "variant",
                [
                    {
                        "title": "LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation",
                        "distance": 0.2,
                    }
                ],
            ),
            (
                "on_topic",
                [
                    {
                        "title": "Federated Sketching LoRA: A Flexible Framework",
                        "distance": 0.25,
                    }
                ],
            ),
        ]
    )
    ranked = list(_rank_grouped_by_title(grouped, q).keys())
    assert ranked[0] == "on_topic"
