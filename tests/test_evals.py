"""Tests for the suite-loading and skip-logic in gene.agent.evals.

The `run` function itself hits the network — not covered here. What's
covered: load_suite returns a Suite with the right shape (across both
roots); skip_reason returns None when there's no TAG and a message when
the family DB isn't built; list_suites finds suites from both roots.
"""

from unittest.mock import patch

from gene.agent.evals import Suite, list_suites, load_suite, skip_reason


def test_load_suite_wraps_module_exports():
    suite = load_suite("basic")
    assert isinstance(suite, Suite)
    assert suite.name == "basic"
    assert suite.tag is None
    assert len(suite.cases) >= 1


def test_load_suite_resolves_genealogy_namespace():
    suite = load_suite("genealogy/david_ancestors")
    assert suite.name == "genealogy/david_ancestors"
    assert suite.tag == "david_ancestors"
    assert len(suite.cases) >= 1


def test_list_suites_includes_both_roots():
    suites = list_suites()
    assert "basic" in suites
    assert "genealogy/david_ancestors" in suites


def test_skip_reason_none_when_no_tag():
    suite = load_suite("basic")
    assert skip_reason(suite) is None


def test_skip_reason_reports_missing_db(tmp_path):
    """Point get_db_path at an empty tmp dir so the tag exists in code but the DB doesn't."""
    suite = load_suite("genealogy/david_ancestors")
    fake = tmp_path / "david_ancestors.sqlite"  # does not exist
    with patch("gene.agent.evals.get_db_path", return_value=fake):
        reason = skip_reason(suite)
    assert reason is not None
    assert "david_ancestors" in reason
    assert "not built" in reason


def test_skip_reason_none_when_db_present(tmp_path):
    suite = load_suite("genealogy/david_ancestors")
    fake = tmp_path / "david_ancestors.sqlite"
    fake.write_bytes(b"")
    with patch("gene.agent.evals.get_db_path", return_value=fake):
        assert skip_reason(suite) is None
