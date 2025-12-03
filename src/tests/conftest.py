# conftest.py
import pytest
from pathlib import Path
import sys

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.char_language_model import CharLanguageModel
import numpy as np

# -------------------------
# CLI options & marker setup
# -------------------------
# conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (skipped by default)"
    )
    parser.addoption(
        "--runveryslow",
        action="store_true",
        default=False,
        help="run veryslow tests (skipped by default). This implies --runslow."
    )
    parser.addoption(
        "--runall",
        action="store_true",
        default=False,
        help="run all tests (overrides --runslow/--runveryslow)."
    )

#def pytest_configure(config):
    # Register markers so pytest does not warn about unknown marks
#    config.addinivalue_line("markers", "slow: mark test as slow (use --runslow to run)")
#    config.addinivalue_line("markers", "veryslow: mark test as very slow (use --runveryslow to run)")
    # (optional) Add any other custom markers here

def pytest_collection_modifyitems(config, items):
    """Skip slow/veryslow tests depending on CLI flags.

    Precedence:
      --runall: run everything
      --runveryslow: run veryslow and slow
      --runslow: run slow, skip veryslow
      (default): skip both slow and veryslow
    """
    runall = config.getoption("--runall")
    runveryslow = config.getoption("--runveryslow")
    runslow = config.getoption("--runslow")

    # If user asked to run veryslow, treat that as also running slow tests.
    if runveryslow:
        runslow = True

    if runall:
        # no skipping
        return

    skip_veryslow = pytest.mark.skip(reason="skipped veryslow test: use --runveryslow or --runall to run")
    skip_slow = pytest.mark.skip(reason="skipped slow test: use --runslow, --runveryslow or --runall to run")

    for item in items:
        if "veryslow" in item.keywords and not runveryslow:
            item.add_marker(skip_veryslow)
        elif "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)


# -------------------------
# Your fixtures (existing)
# -------------------------
@pytest.fixture
def sample_text_a():
    return b"Hello world! This is sample A. " * 2

@pytest.fixture
def sample_text_b():
    return b"Different sample B: emails, text, and more. " * 2

@pytest.fixture
def small_char_model(tmp_path: Path, sample_text_a):
    # train a small model and save to disk
    path = tmp_path / "char_model.bin"
    m = CharLanguageModel(n=5)
    m.train(sample_text_a)
    m.save(path)
    return path

@pytest.fixture
def two_small_models(tmp_path: Path, sample_text_a, sample_text_b):
    p1 = tmp_path / "m1.bin"
    p2 = tmp_path / "m2.bin"
    m1 = CharLanguageModel(n=5)
    m2 = CharLanguageModel(n=5)
    m1.train(sample_text_a)
    m2.train(sample_text_b)
    m1.save(p1)
    m2.save(p2)
    return p1, p2

@pytest.fixture
def two_short_texts():
    """Two realistic short texts for testing"""
    text1 = b"This is a test email message with some content."
    text2 = b"Another example text that differs from the first one."
    return text1, text2

MIN_EMAILS_FOR_TEST = {
    'test_decoder_real_emails_same_domain': 100,
    'test_decoder_different_domains': 2000,
    'test_decoder_various_lengths': 50,
}