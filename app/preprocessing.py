"""Arabic text preprocessing shared by training and inference.

Normalization only: no stemming/lemmatization so that dialect
morphology (which carries sentiment) is preserved.
"""
import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#")
_TATWEEL = "ـ"
_DIACRITICS_RE = re.compile(r"[ً-ْٰ]")
_NON_WORD_RE = re.compile(r"[^\w\s\U0001F300-\U0001FAFF☀-➿]")
_WS_RE = re.compile(r"\s+")

_ALEF_VARIANTS = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
_CHAR_MAP = {**_ALEF_VARIANTS, "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"}


def normalize_arabic(text: str) -> str:
    """Normalize one Arabic text for the sentiment pipeline.

    Steps: strip URLs/mentions, remove diacritics and tatweel,
    unify alef/ya/taa-marbuta variants, drop punctuation and Latin
    script (the datasets are Arabic-script tweets), collapse spaces.
    Emojis are kept: they carry sentiment in dialect tweets.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub("", text)  # keep hashtag text, drop '#'
    text = _DIACRITICS_RE.sub("", text)
    text = text.replace(_TATWEEL, "")
    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text
