import re
import unicodedata
from bs4 import BeautifulSoup


def strip_html(text: str) -> str:
    """
    Remove HTML tags from text.
    """
    if not isinstance(text, str):
        return ""

    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters using NFKC normalization.
    """
    if not isinstance(text, str):
        return ""

    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text: str) -> str:
    """
    Replace repeated spaces, tabs, and newlines with a single space.
    """
    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    """
    Simple language filter.

    Keeps text if most alphabetic characters are ASCII/English characters.
    """
    if not isinstance(text, str) or not text.strip():
        return False

    letters = [char for char in text if char.isalpha()]

    if not letters:
        return False

    english_letters = [
        char for char in letters
        if ("a" <= char.lower() <= "z")
    ]

    ratio = len(english_letters) / len(letters)

    return ratio >= 0.8


def clean_text(text: str) -> str:
    """
    Apply the complete preprocessing pipeline.
    """
    text = strip_html(text)
    text = normalize_unicode(text)
    text = clean_whitespace(text)

    return text


if __name__ == "__main__":
    sample = """
    <p>Hello     world!</p>

    This   is a     test.
    """

    print("Original:")
    print(sample)

    cleaned = clean_text(sample)

    print("\nCleaned:")
    print(cleaned)

    print("\nEnglish:", is_english(cleaned))