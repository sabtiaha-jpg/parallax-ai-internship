import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import (
    strip_html,
    normalize_unicode,
    clean_whitespace,
    is_english,
    clean_text,
)


def test_strip_html():
    text = "<p>Hello <b>world</b></p>"
    result = strip_html(text)

    assert "Hello" in result
    assert "world" in result
    assert "<p>" not in result
    assert "<b>" not in result


def test_normalize_unicode():
    text = "ＡＢＣ"
    result = normalize_unicode(text)

    assert result == "ABC"


def test_clean_whitespace():
    text = "Hello     world\n\nThis\tis a test."
    result = clean_whitespace(text)

    assert result == "Hello world This is a test."


def test_is_english_true():
    text = "This is a simple English sentence."

    assert is_english(text) is True


def test_is_english_false():
    text = "مرحبا كيف حالك اليوم"

    assert is_english(text) is False


def test_clean_text():
    text = "<p>Hello     world!</p>\n\nThis is a test."

    result = clean_text(text)

    assert result == "Hello world! This is a test."


def test_empty_text():
    assert clean_text("") == ""
    assert is_english("") is False