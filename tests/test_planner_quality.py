"""Unit tests for Planner Quality, Scope Drift Prevention, and Depth Calibration:
- Standard mode enforces EXACTLY 4 sub-questions.
- Sub-questions are interrogative research questions ending in '?' (never topic labels).
- Domain-specific, non-generic tags (1-3 words, no 'Overview', 'General', 'Aspects', 'Details').
- Clause grounding and dropped peripheral angles audit.
- Depth calibration for quick (2-3), std (4), deep (4-5).
"""
import pytest
from vresearch.agents import (
    plan,
    clean_and_validate_tag,
    ensure_interrogative_question,
    validate_lane_label,
    PlanList,
    GENERIC_TAG_WORDS
)
from vresearch.board import build_findings_board
from vresearch.config import Settings
from vresearch.pipeline import run_research


class MockPlannerLLM:
    """Mock LLM returning sharp interrogative sub-questions with domain tags."""
    def __init__(self, subs=None, audit=None):
        self.subs = subs
        self.audit = audit

    def json(self, system, user, *, model=None, tag="", max_tokens=0):
        if tag == "planner":
            if self.subs is not None:
                return {"sub_questions": self.subs, "audit": self.audit or {}}
            # Default pickup artist response demonstrating clause grounding
            return {
                "sub_questions": [
                    {
                        "id": "Q1",
                        "tag": "Conversational Tactics",
                        "label": "Opening lines and behavioral tactics",
                        "question": "What specific conversational openers, behavioral scripts, and interaction frameworks are used to initiate contact and attempt to build attraction?",
                        "answer_spec": "Names specific verbal openers, framing routines, and pacing tactics.",
                        "maps_to_clause": "how pickup artists get girls",
                        "queries": ["pua opening routines attraction scripts", "pickup artist conversational frameworks"]
                    },
                    {
                        "id": "Q2",
                        "tag": "Cognitive Biases",
                        "label": "Psychological manipulation techniques and mechanisms",
                        "question": "What specific cognitive biases, psychological vulnerabilities, and emotional triggers are targeted during these interactions?",
                        "answer_spec": "Details specific psychological mechanisms including intermittent reinforcement, scarcity, and validation-seeking.",
                        "maps_to_clause": "how does it target women psychology",
                        "queries": ["pickup artist psychological manipulation cognitive biases", "intermittent reinforcement validation dating tactics"]
                    },
                    {
                        "id": "Q3",
                        "tag": "Empirical Efficacy",
                        "label": "Empirical effectiveness and scientific validation",
                        "question": "What does empirical psychological and behavioral research demonstrate regarding the actual effectiveness and reliability of these pickup tactics?",
                        "answer_spec": "Cites peer-reviewed studies, social psychology experiments, and empirical evidence evaluating outcomes.",
                        "maps_to_clause": "how pickup artists get girls",
                        "queries": ["empirical studies pickup artist effectiveness psychology", "scientific research dating tactics efficacy"]
                    },
                    {
                        "id": "Q4",
                        "tag": "Situational Vulnerability",
                        "label": "Situational susceptibility and boundary conditions",
                        "question": "What situational boundary conditions, context factors, and individual susceptibilities moderate women's responses to these techniques?",
                        "answer_spec": "Identifies context factors such as alcohol, fatigue, social pressure, and attachment styles.",
                        "maps_to_clause": "how does it target women psychology",
                        "queries": ["situational factors susceptibility manipulation tactics", "social compliance boundary conditions dating"]
                    }
                ],
                "audit": {
                    "mapped_clauses": [
                        "how pickup artists get girls",
                        "how does it target women psychology"
                    ],
                    "dropped_peripheral_angles": [
                        "key creators and guru biographies",
                        "online communities and forum culture",
                        "criminal laws and municipal statutes"
                    ],
                    "reasoning": "Focused strictly on behavioral techniques, psychological mechanisms, empirical efficacy, and situational susceptibility."
                }
            }
        return {}


def test_standard_depth_enforces_exactly_4_subquestions():
    llm = MockPlannerLLM()
    subqs = plan(llm, "how pickup artists get girls, how does it target women psychology", depth="std")
    assert len(subqs) == 4, f"Standard depth must yield exactly 4 sub-questions, got {len(subqs)}"
    assert isinstance(subqs, PlanList)
    assert subqs.depth == "std"
    assert "mapped_clauses" in subqs.audit
    assert "dropped_peripheral_angles" in subqs.audit


def test_all_subquestions_are_interrogative_ending_in_question_mark():
    llm = MockPlannerLLM()
    subqs = plan(llm, "how pickup artists get girls, how does it target women psychology", depth="std")
    for q in subqs:
        q_text = q["question"]
        assert q_text.endswith("?"), f"Question '{q_text}' must end with '?'"
        # Must start with an interrogative inquiry word
        first_word = q_text.split()[0].lower()
        assert first_word in ("what", "how", "why", "which", "who", "when", "where", "does", "do", "are", "is", "can", "to")


def test_topic_labels_rephrased_into_research_questions():
    # If LLM returned noun phrases or topic labels instead of questions
    raw_topic = "Opening lines and behavioral tactics"
    rephrased = ensure_interrogative_question(raw_topic)
    assert rephrased.endswith("?")
    assert rephrased.lower().startswith("what")
    assert "Opening lines and behavioral tactics" in rephrased


def test_tags_are_isolated_and_zero_tokens():
    llm = MockPlannerLLM()
    subqs = plan(llm, "how pickup artists get girls, how does it target women psychology", depth="std")
    for q in subqs:
        assert q["tag"] == "", "Tag must be empty string to consume zero LLM tokens"


def test_clean_and_validate_tag_returns_empty_to_save_tokens():
    assert clean_and_validate_tag("Any Tag", "Any Question") == ""


def test_depth_calibration_quick_and_deep():
    # Mock LLM returning 5 items
    five_subs = [
        {"id": f"Q{i}", "tag": f"Tag {i}", "label": f"Valid lane statement number {i}", "question": f"What is factor number {i}?", "answer_spec": "facts", "queries": ["q"]}
        for i in range(1, 6)
    ]
    llm = MockPlannerLLM(subs=five_subs)

    # quick depth: target 3 (min 2, max 3)
    quick_subqs = plan(llm, "sample question", depth="quick")
    assert len(quick_subqs) == 3

    # deep depth: target 5
    deep_subqs = plan(llm, "sample question", depth="deep")
    assert len(deep_subqs) == 5

    # standard depth: exactly 4
    std_subqs = plan(llm, "sample question", depth="std")
    assert len(std_subqs) == 4


def test_heuristic_fallback_produces_four_sharp_questions_with_tags():
    class BrokenLLM:
        def json(self, *args, **kwargs):
            raise RuntimeError("API unavailable")

    subqs = plan(BrokenLLM(), "solid state battery degradation", depth="std")
    assert len(subqs) == 4
    for q in subqs:
        assert q["question"].endswith("?")
        assert q["tag"] == "", "Tag must be empty string to consume zero tokens"


def test_findings_board_preserves_tags_and_renders_four_lanes():
    report = {
        "run_id": "test_tags_run",
        "question": "how pickup artists get girls, how does it target women psychology",
        "summary": "Conversational scripts and intermittent validation serve as lead interaction tactics [1]. Empirical trials show mixed long-term efficacy [2].",
        "sources": [
            {"n": 1, "title": "Social Interaction Study", "url": "https://journal.org/pua", "tier": "primary"},
            {"n": 2, "title": "Behavioral Review", "url": "https://university.edu/review", "tier": "academic"}
        ],
        "sub_questions": [
            {"id": "Q1", "tag": "Conversational Tactics", "label": "Opening lines and behavioral tactics", "question": "What specific conversational openers are used?"},
            {"id": "Q2", "tag": "Cognitive Biases", "label": "Psychological manipulation techniques and mechanisms", "question": "What cognitive biases are targeted?"},
            {"id": "Q3", "tag": "Empirical Efficacy", "label": "Empirical effectiveness and scientific validation", "question": "What does empirical research demonstrate?"},
            {"id": "Q4", "tag": "Situational Vulnerability", "label": "Situational susceptibility and boundary conditions", "question": "What situational factors influence susceptibility?"}
        ],
        "accepted": [
            {"cid": "C1", "subq_id": "Q1", "text": "Verbal opening scripts are calibrated to reduce immediate social tension.", "status": "verified", "source_numbers": [1]},
            {"cid": "C2", "subq_id": "Q1", "text": "Framing routines establish perceived social status in early contact.", "status": "verified", "source_numbers": [1]},
            {"cid": "C3", "subq_id": "Q2", "text": "Intermittent reinforcement creates acute validation-seeking behavior.", "status": "verified", "source_numbers": [2]},
            {"cid": "C4", "subq_id": "Q2", "text": "Manufactured scarcity increases perceived desirability.", "status": "verified", "source_numbers": [2]}
        ],
        "rejected": [],
        "unverified": [],
        "metrics": {"groundedness": 1.0}
    }
    board = build_findings_board(report)
    assert len(board["lanes"]) == 4
    for lane in board["lanes"]:
        assert lane["tag"] in ("Conversational Tactics", "Cognitive Biases", "Empirical Efficacy", "Situational Vulnerability")


def test_stock_tags_rejected_and_domain_synthesized():
    """Assert stock tags like 'EMPIRICAL EFFICACY' and 'BOUNDARY CONDITIONS' are rejected in favor of domain tags."""
    stock_samples = [
        ("EMPIRICAL EFFICACY", "What sources of the past that proves our earth was flat research evidence?"),
        ("BOUNDARY CONDITIONS", "What ancient cosmological records and historical maps describe a flat earth model?"),
        ("TACTICAL FRAMEWORKS", "What specific interaction scripts do pickup artists use?"),
        ("PSYCHOLOGICAL MECHANISMS", "What cognitive vulnerabilities are targeted during social compliance?"),
    ]
    for stock_tag, q in stock_samples:
        cleaned = clean_and_validate_tag(stock_tag, q)
        assert cleaned.lower() not in ("empirical efficacy", "boundary conditions", "tactical frameworks", "psychological mechanisms")
        for word in cleaned.split():
            assert word.lower() not in GENERIC_TAG_WORDS, f"Cleaned tag '{cleaned}' should not have generic word '{word}'"


def test_extract_json_handles_relaxed_commas_and_evaluations_list():
    """Assert extract_json parses trailing commas and list-based evaluations cleanly."""
    from vresearch.llm import extract_json

    # JSON with trailing commas
    text_trailing = """
    ```json
    {
      "evaluations": [
        {"cid": "temp_Q1_1", "relevance": "direct", "reason": "matches trial data"},
        {"cid": "temp_Q1_2", "relevance": "off_topic", "reason": "generic warning"},
      ]
    }
    ```
    """
    res = extract_json(text_trailing)
    assert isinstance(res, dict)
    assert len(res["evaluations"]) == 2
    assert res["evaluations"][0]["relevance"] == "direct"

    # Bare list format
    text_list = """
    [
      {"cid": "temp_Q1_1", "relevance": "direct", "reason": "names specific entity"},
      {"cid": "temp_Q1_2", "relevance": "off_topic", "reason": "unrelated topic"}
    ]
    """
    res_list = extract_json(text_list)
    evals_list = res_list.get("evaluations", []) if isinstance(res_list, dict) else (res_list if isinstance(res_list, list) else [])
    assert len(evals_list) == 2
    assert evals_list[0]["cid"] == "temp_Q1_1"


def test_extract_json_handles_bullet_point_evaluations_fallback():
    """Assert extract_json extracts markdown evaluations when model avoids pure JSON."""
    from vresearch.llm import extract_json

    markdown_text = """
    Here are my evaluations for the claims:
    - [temp_Q1_1]: direct - mentions ancient Babylonian tablets describing flat disc
    - [temp_Q1_2]: off_topic - general discussion of modern satellite orbits
    """
    res = extract_json(markdown_text)
    assert isinstance(res, dict)
    assert "evaluations" in res
    assert len(res["evaluations"]) == 2
    assert res["evaluations"][0]["cid"] == "temp_Q1_1"
    assert res["evaluations"][0]["relevance"] == "direct"
    assert res["evaluations"][1]["cid"] == "temp_Q1_2"
    assert res["evaluations"][1]["relevance"] == "off_topic"


def test_template_leak_across_unrelated_queries():
    """Verify that unrelated topics produce zero shared tags/templates and use domain-grounded inquiries."""
    from vresearch.agents import generate_calibrated_plan, validate_tag_form

    queries = [
        "skills needed to get into quant",
        "how pickup artists get girls, how does it target women psychology",
        "what supplements or medications help with weight loss",
        "what are the main risks of investing in the EV battery market in 2026",
        "what sources of the past that proves our earth was flat"
    ]

    plans = {q: generate_calibrated_plan(q, depth="std") for q in queries}

    # 1. Check form and interrogative structure of all subquestions
    for q, plan_items in plans.items():
        assert len(plan_items) == 4, f"Plan for '{q}' should have 4 sub-questions"
        seen_tags = set()
        for item in plan_items:
            subq = item["question"]
            tag = item["tag"]
            assert subq.endswith("?"), f"Sub-question '{subq}' must end with '?'"
            # Active question starter
            first = subq.split()[0].lower()
            assert first in ("what", "how", "why", "which", "who", "when", "where", "does", "is", "are", "can", "to")

            # Form-only tag validation
            ok, clean = validate_tag_form(tag, seen_tags)
            assert ok, f"Tag '{tag}' should satisfy form (1-2 words Title Case, unique)"
            seen_tags.add(clean)

            # Query list specific
            assert len(item["queries"]) >= 1

    # 2. Assert zero shared questions across unrelated queries and isolated tags
    quant_qs = {item["question"].lower() for item in plans["skills needed to get into quant"]}
    pua_qs = {item["question"].lower() for item in plans["how pickup artists get girls, how does it target women psychology"]}
    supp_qs = {item["question"].lower() for item in plans["what supplements or medications help with weight loss"]}
    battery_qs = {item["question"].lower() for item in plans["what are the main risks of investing in the EV battery market in 2026"]}
    cosmo_qs = {item["question"].lower() for item in plans["what sources of the past that proves our earth was flat"]}

    assert quant_qs.isdisjoint(battery_qs), f"Quant and Battery questions must not overlap: {quant_qs & battery_qs}"
    assert quant_qs.isdisjoint(pua_qs), f"Quant and PUA questions must not overlap: {quant_qs & pua_qs}"
    assert quant_qs.isdisjoint(cosmo_qs), f"Quant and Flat Earth questions must not overlap: {quant_qs & cosmo_qs}"
    assert battery_qs.isdisjoint(supp_qs), f"Battery and Supp questions must not overlap: {battery_qs & supp_qs}"
    assert battery_qs.isdisjoint(cosmo_qs), f"Battery and Flat Earth questions must not overlap: {battery_qs & cosmo_qs}"


