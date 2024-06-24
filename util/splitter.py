"""This module contains functions for splitting text into chunks of roughly equal length."""

import nltk

_nltk_is_initialized = False


def split_text(text: str, chunk_len: int = 256) -> list[str]:
    """Split text into semantic segments of roughly equal length.

    :param text: The text to split.
    :param chunk_len: The target length of each chunk.
    :return: A list of chunks of text.
    """
    global _nltk_is_initialized
    if not _nltk_is_initialized:
        _nltk_is_initialized = True
        nltk.download("punkt")
    # first use nltk to split into sentences
    sentences: list[str] = nltk.tokenize.sent_tokenize(text)  # type: ignore
    # split each sentence into chunks of target length if it's not already below the target length
    clean_sentences: list[str] = []
    for i, sentence in enumerate(sentences):
        if len(sentence) < chunk_len:
            clean_sentences.append(sentence)
        else:
            # use nltk to split the sentence into words
            words: list[str] = nltk.tokenize.word_tokenize(sentence)  # type: ignore
            chunk: list[str] = []
            for word in words:
                # FIXME: there is a bug here, chunk is a list of strings and word a string, adding the to metrics makes no sense
                if len(chunk) + len(word) < chunk_len:
                    chunk.append(word)
                else:
                    clean_sentences.append(" ".join(chunk))  # TODO: this isn't qutie rigth, need to de-tokenize instead
                    chunk = [word]
            if len(chunk) > 0:
                clean_sentences.append(" ".join(chunk))
    # now, let's merge neighboring chunks that are below the target length
    i = 0
    while i < len(clean_sentences) - 1:
        if len(clean_sentences[i]) + len(clean_sentences[i + 1]) < chunk_len:
            clean_sentences[i] = clean_sentences[i] + " " + clean_sentences[i + 1]
            del clean_sentences[i + 1]
        else:
            i += 1
    return [s for s in clean_sentences if len(s) < chunk_len]
