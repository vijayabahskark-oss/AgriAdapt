import re
import unicodedata


STATE_ALIASES = {
    "ORISSA": "ORISSA",
    "ODISHA": "ORISSA",

    "UTTARANCHAL": "UTTARANCHAL",
    "UTTARAKHAND": "UTTARANCHAL",

    "PONDICHERRY": "PONDICHERRY",
    "PUDUCHERRY": "PONDICHERRY",

    "JAMMU KASHMIR": "JAMMU AND KASHMIR",
    "JAMMU AND KASHMIR": "JAMMU AND KASHMIR",

    "A N ISLANDS": "A N ISLANDS",
    "A AND N ISLANDS": "A N ISLANDS",
    "ANDAMAN AND NICOBAR ISLANDS":
        "A N ISLANDS",

    "D N HAVELI": "D N HAVELI",
    "DADRA AND NAGAR HAVELI":
        "D N HAVELI",
}


def normalize_text(value):
    if value is None:
        return ""

    text = str(value).strip()

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = "".join(
        char
        for char in text
        if not unicodedata.combining(char)
    )

    text = text.upper()

    text = text.replace(
        "&",
        " AND "
    )

    text = re.sub(
        r"[^A-Z0-9]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def normalize_state(value):
    text = normalize_text(value)

    return STATE_ALIASES.get(
        text,
        text
    )


def normalize_district(value):
    return normalize_text(value)