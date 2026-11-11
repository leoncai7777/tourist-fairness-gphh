"""
A check of the self-check itself: every deliberate bug of tools/self_check.py must find its text exactly
once in the code, or it silently stops checking anything. Runs with the quick tests, so a change of the
code that breaks a bug's text is seen at once, not only at the end of a full self-check.

Inside the self-check's own copies (src and tests only, no tools) there is nothing to check: skipped.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "self_check.py"


@pytest.mark.skipif(not TOOL.exists(), reason="inside a copy made by the self-check")
def test_every_deliberate_bug_finds_its_text_exactly_once():
    sys.path.insert(0, str(TOOL.parent))
    import self_check
    wrong = [(meaning, (ROOT / path).read_text().count(text))
             for path, text, bug, meaning in self_check.MUTATIONS if (ROOT / path).read_text().count(text) != 1]
    assert wrong == []
