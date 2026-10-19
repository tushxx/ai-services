import csv
from pathlib import Path
import re
import numpy as np
import pandas as pd

from config import constants
from metermind.message_intelligence.attention import attention_weights
from metermind.message_intelligence.text_cleaning import clean_message

SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])\s+")
SHORTEST_SENTENCE_CHARACTERS = 20


class UnreadableThread(Exception):
    """Raised where a thread carries too little to summarise."""
    pass


def split_sentences(thread_text):
    if not thread_text or not str(thread_text).strip():
        return []
    pieces = SENTENCE_END_PATTERN.split(str(thread_text).strip())
    sentences = []
    for piece in pieces:
        cleaned = re.sub(r"\s+", " ", piece).strip()
        if len(cleaned) >= SHORTEST_SENTENCE_CHARACTERS:
            sentences.append(cleaned)
    return sentences


def sentence_matrix(sentences):
    vocab_set = set()
    sentence_tokens = []
    for s in sentences:
        words = [re.sub(r"[^\w]+$", "", w.lower()).strip() for w in s.split()]
        words = [w for w in words if w]
        sentence_tokens.append(words)
        vocab_set.update(words)
    vocab = sorted(vocab_set)
    if not vocab:
        return np.zeros((len(sentences), 0), dtype=float)
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(sentences), len(vocab)), dtype=float)
    for i, tokens in enumerate(sentence_tokens):
        for token in tokens:
            if token in word_to_idx:
                mat[i, word_to_idx[token]] += 1.0
    return mat


def score_sentences(sentences):
    if len(sentences) < 2:
        raise UnreadableThread("A thread of fewer than two sentences has nothing to compare.")
    mat = sentence_matrix(sentences)
    if mat.shape[1] == 0:
        raise UnreadableThread("No vocabulary found in sentences.")
    weights = attention_weights(mat, mat)
    # Sum over axis 0 (total attention directed at each sentence)
    scores = np.asarray(weights.sum(axis=0), dtype=float)
    return scores


def summarise_thread(thread_text, sentence_count=None):
    if sentence_count is None:
        sentence_count = constants.SUMMARY_SENTENCE_COUNT
    sentences = split_sentences(thread_text)
    if not sentences:
        raise UnreadableThread("A thread with no sentence long enough cannot be summarised.")
    if len(sentences) <= sentence_count:
        return " ".join(sentences)
    scores = score_sentences(sentences)
    top_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:sentence_count]
    top_indices.sort()
    chosen = [sentences[i] for i in top_indices]
    return " ".join(chosen)


def write_thread_summaries(threads_path, summaries_path):
    p = Path(threads_path)
    if not p.exists():
        raise UnreadableThread(f"Threads file not found: {threads_path}")
    df = pd.read_csv(threads_path)
    summaries = []
    total_threads = 0
    summarised_count = 0
    not_summarised_count = 0

    for thread_id, group in df.groupby("thread_id", sort=False):
        total_threads += 1
        turns = len(group)
        cleaned_turns = [clean_message(str(msg)) for msg in group["message_text"]]
        joined_text = " ".join([t for t in cleaned_turns if t])
        try:
            summary = summarise_thread(joined_text)
            summarised_count += 1
        except UnreadableThread as err:
            summary = str(err)
            not_summarised_count += 1
        summaries.append({
            "thread_id": thread_id,
            "turns": turns,
            "summary": summary,
        })

    out = Path(summaries_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["thread_id", "turns", "summary"])
        writer.writeheader()
        writer.writerows(summaries)

    return {
        "threads": total_threads,
        "summarised": summarised_count,
        "not summarised": not_summarised_count,
    }


if __name__ == "__main__":
    result = write_thread_summaries(constants.MESSAGE_THREADS_PATH, constants.THREAD_SUMMARIES_PATH)
    print(f"written: {result}")
