"""Reversible repairs of text-extraction damage in the KB sources, checked against the corpus vocabulary.

vocabulary(texts) -> {word: document frequency} of lowercase Cyrillic words of 4+ letters and hyphenated compounds
  ("какие-то"), kept when found in min_df documents or more.
repair_encoding(text, vocab) -> (text, info or None): undoes a 1:1 character-level mis-decoding of the whole document
  (cp1251 bytes read as Latin-1: "ÌÈÍÈÑÒÅÐÑÒÂÎ"; a PDF font whose glyph codes are Cyrillic moved by a constant:
  "ǳȍȒȞȐȐ" = "Лекции"), accepted when most decoded words are vocabulary words, several different ones. Letters the
  constant leaves wrong (fonts often place capitals or single letters elsewhere) get the letter that turns the most
  words containing them into vocabulary words. The length is kept, so offsets hold.
index_form(text, vocab) -> (text, counts): the form of a chunk for the search index (BM25 and embeddings); the text shown
  to readers keeps the source wording. Words broken at a line or page wrap ("сен- тября", "инфор-⏎мацию") are joined
  when the joined word is a vocabulary word more frequent than the hyphenated compound; letter-spaced runs of 6+ letters
  ("М И Н И С Т Е Р С Т В О", "Р у с с к и й я з ы к") are joined into vocabulary words of 4+ letters."""

import collections
import itertools
import re

WORD = re.compile(r"[а-яё]{4,}")
COMPOUND = re.compile(r"(?<![а-яё-])[а-яё]+(?:-[а-яё]+)+")
# a word broken by a hyphen followed by a space or a line break (a blank line too: a page break)
BROKEN = re.compile(r"(?<![\w-])[А-Яа-яЁё]{2,}(?:-(?:[ \t]+|[ \t]*\n[ \t]*\n?[ \t]*)[А-Яа-яЁё]{2,})+")
PIECE = re.compile(r"[А-Яа-яЁё]+")
SPACED = re.compile(r"(?<!\S)(?:[А-Яа-яЁё] ){5,}[А-Яа-яЁё](?!\S)")
SEG_MIN = 4
MULTI_DF = 30  # a run split into several words: each must be this common (rare ones are fragments of other words)

LATIN1 = re.compile(r"(?<![A-Za-z])[¨¸¹À-ÿ]+(?![A-Za-z])")  # a word of Latin-1 letters only
CP1251 = {c: c.encode("cp1252").decode("cp1251") for c in "¨¸¹" + "".join(map(chr, range(0xC0, 0x100)))}
SHIFTABLE = re.compile(r"[ƀ-Ͽ]+")  # Latin Extended-B .. Greek: where shifted Cyrillic glyph codes land
SUSPECT = re.compile(r"[¨¸¹À-ÿƀ-Ͽ]")
MIN_WORDS, MIN_DISTINCT, MIN_KNOWN = 5, 5, 0.5
PRE_KNOWN = 0.2  # a shift is refined when this share of its words is known, accepted at MIN_KNOWN after refining
RU = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
RU_SET = set(RU)
EDGE = ".,;:!?()[]{}\"'«»—–-…"
REFINE_CHARS, REFINE_TOKENS = 90, 300
SOURCE_LETTER_SHARE = 0.5  # a Russian letter of the source is reassigned only when that makes most of its words known


def vocabulary(texts, min_df=3):
    df = collections.Counter()
    for t in texts:
        low = t.lower()
        df.update(set(WORD.findall(low)) | set(COMPOUND.findall(low)))
    return {w: n for w, n in df.items() if n >= min_df}


# ---- index form


def _continues(prev, piece):
    """piece may continue the broken word prev: lowercase, or upper case after an all-caps start"""
    return piece.islower() or (piece.isupper() and prev.isupper())


def _join_broken(m, vocab, counts):
    s = m.group()
    pieces = [s[x.start() : x.end()] for x in PIECE.finditer(s)]
    seps = [x.group() for x in re.finditer(r"[^А-Яа-яЁё]+", s)]  # the "- " or "-⏎" between pieces
    out, i = [], 0
    while i < len(pieces):
        j = i
        for k in range(len(pieces) - 1, i, -1):
            if not all(_continues(pieces[x - 1], pieces[x]) for x in range(i + 1, k + 1)):
                continue
            df = vocab.get("".join(pieces[i : k + 1]).lower(), 0)
            compound = vocab.get("-".join(pieces[i : k + 1]).lower(), 0)
            if df > compound and vocab.get(pieces[k].lower(), 0) < df:
                j = k
                counts["dehyphenated"] += 1
                break
        out.append("".join(pieces[i : j + 1]))
        if j + 1 < len(pieces):
            out.append(seps[j])
        i = j + 1
    return "".join(out)


def _segments(word, vocab):
    """split word into vocabulary words of SEG_MIN+ letters, maximising the least document frequency; None if impossible"""
    low = word.lower()
    best = {0: (float("inf"), [])}  # prefix length -> (least df, cut positions)
    for e in range(SEG_MIN, len(low) + 1):
        for s in range(e - SEG_MIN + 1):
            if s in best and low[s:e] in vocab:
                cand = (min(best[s][0], vocab[low[s:e]]), [*best[s][1], e])
                if e not in best or cand[0] > best[e][0]:
                    best[e] = cand
    if len(low) not in best or (len(best[len(low)][1]) > 1 and best[len(low)][0] < MULTI_DF):
        return None
    cuts = [0, *best[len(low)][1]]
    return [word[a:b] for a, b in itertools.pairwise(cuts)]


def _join_spaced(m, vocab, counts):
    segs = _segments(m.group().replace(" ", ""), vocab)
    if segs:
        counts["letter_spacing_joined"] += 1
        return " ".join(segs)
    return m.group()


def index_form(text, vocab):
    counts = collections.Counter()
    text = BROKEN.sub(lambda m: _join_broken(m, vocab, counts), text)
    text = SPACED.sub(lambda m: _join_spaced(m, vocab, counts), text)
    return text, dict(counts)


# ---- encoding


def _known(words, vocab):
    """(decoded words of 4+ letters, of them vocabulary words, distinct vocabulary words)"""
    n, k, seen = 0, 0, set()
    for w in words:
        for x in WORD.findall(w.lower()):
            n += 1
            if x in vocab:
                k += 1
                seen.add(x)
    return n, k, len(seen)


def _accept(n, k, d):
    return n >= MIN_WORDS and d >= MIN_DISTINCT and k >= MIN_KNOWN * n


def _shift_map(s):
    return {chr(c): chr(c + s) for c in range(max(0x180, 0x400 - s), min(0x400, 0x460 - s))}


def _decode(token, cmap):
    return "".join(cmap.get(c, c) for c in token)


def _score(tokens, cmap, vocab):
    """tokens whose decoded form, without edge punctuation, is a vocabulary word"""
    return sum(_decode(t, cmap).strip(EDGE).lower() in vocab for t in tokens)


def _capital(c, tokens, cmap):
    """c is a capital letter: it follows capitals or starts words of lowercase letters more often than it follows
    lowercase letters (the vocabulary is lowercase, so the case comes from the neighbours)"""
    up = low = 0
    for t in tokens:
        d = _decode(t, cmap)
        for i, ch in enumerate(t):
            if ch == c:
                prev, nxt = (d[i - 1] if i else ""), (d[i + 1] if i + 1 < len(d) else "")
                if prev in RU_SET:
                    up += prev.isupper()
                    low += prev.islower()
                elif nxt in RU_SET and nxt.islower():
                    up += 1
    return up > low


def _refine(tokens, cmap, vocab):
    """give each non-ASCII character of the shifted words the letter that makes the most vocabulary words, when that
    is more than its current letter makes (a capital moved by another constant may land on a valid lowercase letter);
    a Russian letter of the source (fonts code "." as "й" or "П" as "А") only when most of its words become known"""
    cmap = dict(cmap)
    for _ in range(2):
        odd = collections.Counter(c for t in tokens for c in t if not c.isascii())
        changed = False
        for c, _n in odd.most_common(REFINE_CHARS):
            near = [t for t in tokens if c in t][:REFINE_TOKENS]
            base = _score(near, cmap, vocab)
            scores = [(_score(near, {**cmap, c: t}, vocab), t) for t in RU]
            top = max(x for x, _ in scores)
            if top > base and (c not in RU_SET or top >= SOURCE_LETTER_SHARE * len(near)):
                ties = [t for x, t in scores if x == top]
                t = ties[0]
                if t.swapcase() in ties:
                    t = t.upper() if _capital(c, near, cmap) else t.lower()
                cmap[c] = t
                changed = True
        if not changed:
            break
    return cmap


def repair_encoding(text, vocab):
    if not SUSPECT.search(text):
        return text, None
    best = None  # (known words, name, char map)
    words = LATIN1.findall(text)[:2000]
    n, k, d = _known((_decode(t, CP1251) for t in words), vocab)
    if _accept(n, k, d):
        best = (k, "cp1251_as_latin1", CP1251)
    runs = SHIFTABLE.findall(text)
    common = collections.Counter(c for r in runs for c in r).most_common(4)
    for s in sorted({ord(t) - ord(c) for c, _ in common for t in "оеаи"}):
        shift = _shift_map(s)
        n, k, d = _known((_decode(r, shift) for r in runs[:2000]), vocab)
        if d < MIN_DISTINCT or k < PRE_KNOWN * n:
            continue
        tokens = [t for t in re.findall(r"\S+", text) if any(c in shift for c in t)][:3000]
        cmap = _refine(tokens, shift, vocab)
        n, k, d = _known((_decode(t, cmap) for t in tokens), vocab)
        if _accept(n, k, d) and (best is None or k > best[0]):
            best = (k, f"shift_{s:#x}", cmap, shift)
    if best is None:
        return text, None
    k, name, cmap = best[:3]
    if name == "cp1251_as_latin1":
        out = LATIN1.sub(lambda m: _decode(m.group(), CP1251), text)
    else:  # only inside words that contain shifted characters: elsewhere « or ° keep their meaning
        shift = best[3]
        out = re.sub(
            r"\S+", lambda m: _decode(m.group(), cmap) if any(c in shift for c in m.group()) else m.group(), text
        )
    return out, {"map": name, "chars": sum(a != b for a, b in zip(text, out, strict=True)), "known_words": k}
