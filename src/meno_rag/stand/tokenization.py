import re
from typing import Optional

from nltk import wordpunct_tokenize
from nltk.stem.snowball import SnowballStemmer
from razdel import tokenize


def tokenize_and_normalize_text(s: str, stemmer: Optional[SnowballStemmer] = None) -> str:
    re_for_digit = re.compile(r"\d+")
    new_tokens: list[str] = []
    for cur in tokenize(s):
        if re_for_digit.search(cur.text) is None:
            new_tokens += list(
                filter(
                    lambda x2: (len(x2) > 0) and x2.isalpha(),
                    map(lambda x1: x1.strip().lower(), wordpunct_tokenize(cur.text)),
                )
            )
        else:
            token_text = cur.text.lower().strip()
            if len(token_text) > 0:
                new_tokens.append(token_text)
    if stemmer is None:
        stemmed_tokens = new_tokens
    else:
        stemmed_tokens = list(
            filter(
                lambda it2: len(it2) > 0,
                map(lambda it1: stemmer.stem(it1).strip(), new_tokens),
            )
        )
    if len(stemmed_tokens) == 0:
        return ""
    return " ".join(stemmed_tokens)
