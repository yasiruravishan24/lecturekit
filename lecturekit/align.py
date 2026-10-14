"""Align speech segments to slides (monotone, TF-IDF similarity) and fix
slide-term errors in the transcript."""
import difflib
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_']*")


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def align(segments: list[dict], slides: list[dict], jump_penalty: float = 0.05) -> list[int]:
    """Return a slide index (0-based) per segment, non-decreasing in time.

    Dynamic programming: maximise total cosine similarity minus a penalty for
    each change of slide, so the lecture moves forward through the slides.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not segments or not slides:
        return [0] * len(segments)
    docs = [s["text"] for s in slides] + [g["text"] for g in segments]
    tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs)
    sl, sg = tfidf[:len(slides)], tfidf[len(slides):]
    sim = (sg @ sl.T).toarray()                       # (segments, slides)
    n, m = sim.shape
    dp = np.full((n, m), -1e9)
    back = np.zeros((n, m), dtype=int)
    dp[0] = sim[0]
    for i in range(1, n):
        best, best_j = -1e9, 0
        for j in range(m):
            if dp[i - 1, j] > best:                   # best predecessor with k <= j
                best, best_j = dp[i - 1, j], j
            stay = dp[i - 1, j]
            move = best - jump_penalty
            if stay >= move:
                dp[i, j], back[i, j] = stay + sim[i, j], j
            else:
                dp[i, j], back[i, j] = move + sim[i, j], best_j
    j = int(dp[-1].argmax())
    path = [j]
    for i in range(n - 1, 0, -1):
        j = int(back[i, j])
        path.append(j)
    return path[::-1]


def correct_with_slides(segments: list[dict], slides: list[dict],
                        cutoff: float = 0.86) -> tuple[list[dict], list[tuple[str, str]]]:
    """Replace rare transcript words that look like a slide term (a misheard
    technical term such as 'itteration' for 'iteration') with that term.

    Conservative rule: the word is not on any slide, occurs at most twice in the
    transcript, is >= 5 letters, shares its first letter with the slide term and has
    a similarity >= cutoff. Returns the corrected segments and the changes made.
    """
    vocab = Counter(w for s in slides for w in words(s["text"]) if len(w) >= 5)
    freq = Counter(w for g in segments for w in words(g["text"]))
    changes, out = [], []
    for g in segments:
        def fix(m):
            w = m.group(0)
            lw = w.lower()
            if len(lw) < 5 or lw in vocab or freq[lw] > 2 or "'" in lw:
                return w
            cand = difflib.get_close_matches(lw, [v for v in vocab if v[0] == lw[0]], 1, cutoff)
            # never swap inflections (lecture/lectures, matter/matters): not ASR errors
            cand = [v for v in cand if not (v.startswith(lw) or lw.startswith(v))]
            if not cand:
                return w
            changes.append((w, cand[0]))
            return cand[0].capitalize() if w[0].isupper() else cand[0]
        out.append({**g, "text": _WORD.sub(fix, g["text"])})
    return out, changes
