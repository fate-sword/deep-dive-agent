import pytest
from vresearch.board import build_findings_board


@pytest.fixture
def sample_report():
    return {
        "question": "What are the main risks of investing in the EV battery market in 2026?",
        "summary": "Raw material supply bottlenecks and recycling compliance rules remain primary investment risks in 2026 [1][2]. Strong cell demand continues [3].",
        "sub_questions": [
            {"id": "Q1", "question": "What raw material supply chain risks exist in 2026?"},
            {"id": "Q2", "question": "What regulatory recycling standards apply in 2026?"},
            {"id": "Q3", "question": "What solid-state commercialization timeline risks exist?"},
        ],
        "accepted": [
            {
                "cid": "C1",
                "subq_id": "Q1",
                "text": "Lithium and cobalt refining concentration creates severe supply chain disruption vulnerabilities.",
                "evidence_ids": ["E1"],
                "source_numbers": [1],
                "status": "verified",
            },
            {
                "cid": "C2",
                "subq_id": "Q1",
                "text": "Battery pack costs declined to $115 per kWh in late 2025.",
                "evidence_ids": ["E2"],
                "source_numbers": [2],
                "status": "verified",
            },
            {
                "cid": "C3",
                "subq_id": "Q2",
                "text": "Regional extended producer responsibility mandates require 70% lithium recovery by 2030.",
                "evidence_ids": ["E3"],
                "source_numbers": [3],
                "status": "verified",
            },
            {
                "cid": "C4",
                "subq_id": "Q2",
                "text": "Failure to meet recycled content minimums incurs commercial distribution restrictions.",
                "evidence_ids": ["E4"],
                "source_numbers": [3],
                "status": "verified",
            },
            # Note: Q3 has only 1 verified claim -> should be marked thin ("not enough proof found")
            {
                "cid": "C5",
                "subq_id": "Q3",
                "text": "Solid-state cell volume production has been delayed past 2028 by major OEMs.",
                "evidence_ids": ["E5"],
                "source_numbers": [4],
                "status": "verified",
            },
        ],
        "unverified": [
            {
                "cid": "C6",
                "subq_id": "Q1",
                "text": "Lithium spot prices will increase by 45% in Q3 2026.",
                "evidence_ids": ["E1"],
                "status": "unverified",
                "reason": "predictive projection not stated in cited historical report",
            }
        ],
        "rejected": [
            {
                "cid": "C7",
                "subq_id": "Q2",
                "text": "All battery recycling was banned in Europe.",
                "evidence_ids": [],
                "status": "rejected",
                "reason": "cites no retrievable source and contradicts EU directive",
            }
        ],
        "sources": [
            {"n": 1, "url": "https://iea.org/reports/ev-outlook", "title": "IEA Report"},
            {"n": 2, "url": "https://bloomberg.com/news/battery-costs", "title": "BloombergNEF"},
            {"n": 3, "url": "https://europa.eu/battery-directive", "title": "EU Battery Directive"},
            {"n": 4, "url": "https://reuters.com/solid-state-delay", "title": "Reuters"},
        ],
        "metrics": {"groundedness": 0.83, "coverage": 1.0, "loops": 0},
    }


def test_notes_text_strictly_equals_accepted_claims(sample_report):
    """Assert every note's text exactly equals an accepted claim's text verbatim."""
    board = build_findings_board(sample_report)
    accepted_by_cid = {c["cid"]: c for c in sample_report["accepted"]}

    total_notes = 0
    for lane in board["lanes"]:
        for note in lane["notes"]:
            total_notes += 1
            cid = note["cid"]
            assert cid in accepted_by_cid, f"Note cid {cid} not found in accepted claims!"
            accepted_claim = accepted_by_cid[cid]

            # Verbatim text equality check
            assert note["text"] == accepted_claim["text"], (
                f"Note text '{note['text']}' does not match accepted claim text '{accepted_claim['text']}'!"
            )
            assert note["status"] == "verified"
            assert note["status_label"] == "Verified"

    assert total_notes == len(sample_report["accepted"])


def test_no_rejected_or_unverified_claims_rendered_as_notes(sample_report):
    """Assert that no rejected or unverified claim is EVER rendered as a note on the board."""
    board = build_findings_board(sample_report)
    unverified_texts = {c["text"] for c in sample_report["unverified"]}
    rejected_texts = {c["text"] for c in sample_report["rejected"]}

    for lane in board["lanes"]:
        for note in lane["notes"]:
            assert note["text"] not in unverified_texts, f"Unverified claim rendered as note: {note['text']}"
            assert note["text"] not in rejected_texts, f"Rejected claim rendered as note: {note['text']}"
            assert note["status"] == "verified"

    # Assert they appear in corner tag
    corner_tag = board["corner_tag"]
    assert corner_tag["rejected_count"] == 1
    assert corner_tag["unverified_count"] == 1
    assert "1 rejected, 1 unverified" in corner_tag["label"]

    assert corner_tag["rejected"][0]["cid"] == "C7"
    assert "contradicts EU directive" in corner_tag["rejected"][0]["reason"]

    assert corner_tag["unverified"][0]["cid"] == "C6"
    assert "predictive projection" in corner_tag["unverified"][0]["reason"]


def test_thin_lanes_say_not_enough_proof_found(sample_report):
    """A lane with fewer than 2 verified claims shows 'not enough proof found' in its header."""
    board = build_findings_board(sample_report)
    lanes_by_id = {l["subq_id"]: l for l in board["lanes"]}

    # Q1 has 2 claims -> not thin
    assert not lanes_by_id["Q1"]["is_thin"]
    assert lanes_by_id["Q1"]["thin_label"] == ""

    # Q2 has 2 claims -> not thin
    assert not lanes_by_id["Q2"]["is_thin"]
    assert lanes_by_id["Q2"]["thin_label"] == ""

    # Q3 has 1 claim (< 2) -> thin
    assert lanes_by_id["Q3"]["is_thin"]
    assert lanes_by_id["Q3"]["thin_label"] == "not enough proof found"


def test_headline_finding_and_footer(sample_report):
    board = build_findings_board(sample_report)
    assert board["mono_label"] == "KEY FINDINGS · PINNED FROM VERIFIED CLAIMS"
    assert "Raw material supply bottlenecks" in board["headline"]["text"]
    assert bool(board["headline"]["highlight"])
    assert board["footer"]["grounded_pct"] == 83
