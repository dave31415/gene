"""Tests for suite discovery and skip-logic in gene.agent.evals.

The runner is domain-agnostic: it takes a directory and duck-types the
modules inside. Covered here: `list_suites(dir)` finds the modules,
`load_suite(dir, name)` wraps their exports in a Suite, and
`skip_reason` runs the suite's `precheck` when present. `run` hits the
network, so it's not tested here.
"""

from unittest.mock import patch

from gene.agent.evals import Suite, list_suites, load_suite, skip_reason

AGENT_CASES = "gene/agent/eval_cases"
GENEALOGY_CASES = "gene/genealogy/eval_cases"


def test_list_suites_finds_agent_cases():
    assert "basic" in list_suites(AGENT_CASES)


def test_list_suites_finds_genealogy_cases():
    assert "david_ancestors" in list_suites(GENEALOGY_CASES)


def test_load_suite_wraps_agent_module_exports():
    suite = load_suite(AGENT_CASES, "basic")
    assert isinstance(suite, Suite)
    assert suite.name == "basic"
    assert suite.precheck is None
    assert len(suite.cases) >= 1


def test_load_suite_wraps_genealogy_module_exports():
    suite = load_suite(GENEALOGY_CASES, "david_ancestors")
    assert suite.name == "david_ancestors"
    assert suite.build_conversation is not None
    assert suite.precheck is not None
    assert len(suite.cases) >= 1


def test_skip_reason_none_when_no_precheck():
    suite = load_suite(AGENT_CASES, "basic")
    assert skip_reason(suite) is None


def test_skip_reason_reports_missing_db(tmp_path):
    """Point get_db_path at an empty tmp dir so the family_tag exists in code but the DB doesn't."""
    suite = load_suite(GENEALOGY_CASES, "david_ancestors")
    fake = tmp_path / "david_ancestors.sqlite"  # does not exist
    with patch("gene.genealogy.eval_cases.david_ancestors.get_db_path", return_value=fake):
        reason = skip_reason(suite)
    assert reason is not None
    assert "david_ancestors" in reason
    assert "not built" in reason


def test_skip_reason_none_when_db_present(tmp_path):
    suite = load_suite(GENEALOGY_CASES, "david_ancestors")
    fake = tmp_path / "david_ancestors.sqlite"
    fake.write_bytes(b"")
    with patch("gene.genealogy.eval_cases.david_ancestors.get_db_path", return_value=fake):
        assert skip_reason(suite) is None
