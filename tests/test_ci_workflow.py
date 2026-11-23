"""The "All green" job in ci.yml is the required check, so it must wait for every job."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CI = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def test_all_green_needs_every_other_job():
    jobs = yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]
    gate = jobs["all-green"]
    assert gate["name"] == "All green"
    assert gate["if"] == "always()"
    assert sorted(gate["needs"]) == sorted(j for j in jobs if j != "all-green")


def test_every_job_has_a_timeout():
    # A hung step (apt-get once sat for 54 minutes) otherwise holds a job for the
    # 6 h default and blocks the required check.
    jobs = yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]
    missing = [name for name, job in jobs.items() if "timeout-minutes" not in job]
    assert not missing, f"jobs without timeout-minutes: {missing}"
