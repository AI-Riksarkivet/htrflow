"""
Token alignment utilities.

This module contains functions to align token IDs with characters and
words in the decoded token sequence. It does basically the same thing as
the `encode_plus` method of PreTrainedTokenizerBase, but can be used on
predicted (rather than encoded) token sequences.
"""

import re

from transformers import PreTrainedTokenizerBase, TokenSpan


def align_chars_to_tokens(token_ids: list[int], tokenizer: PreTrainedTokenizerBase) -> list[TokenSpan]:
    """
    Align characters to tokens

    Computes the alignment between characters of the decoded token_ids to
    their respective token(s). This is a many-to-many relationship: a token
    may consist of several characters, but, a single character can also be
    represented by several tokens. An example of the latter case is ByT5Tokenizer's
    encoding of non-ascii characters.

    Inputs:
        token_ids: List of input token IDs.
        tokenizer: The tokenizer.

    Returns:
        A list of TokenSpan instances that map each character of the decoded
        token sequence to its corresponding token(s). Special tokens are ignored.
    """

    chars = []
    i = 0
    while i < len(token_ids):
        # Skip special IDs
        if token_ids[i] in tokenizer.all_special_ids:
            i += 1
            continue

        # Find the end index of the current character
        j = i + 1
        token = ""
        while 1:
            token = tokenizer.decode(token_ids[i:j])
            if token:
                break
            j += 1

        for _ in token:
            chars.append(TokenSpan(i, j))

        i = j

    return chars


def align_words_to_tokens(token_ids: list[int], tokenizer: PreTrainedTokenizerBase) -> list[TokenSpan]:
    """
    Align words to tokens

    Computes the alignment between words of the decoded token_ids to their
    respective token(s).

    Arguments:
        token_ids: Input token IDs.
        tokenizer: The tokenizer.

    Returns:
        A list of TokenSpans that map each word of the decoded token sequence
        to the indices to its corresponding tokens. Special tokens are ignored.
    """
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    alignment = align_chars_to_tokens(token_ids, tokenizer)

    words = []
    for match in re.finditer(r"\S+", decoded_text):
        start = alignment[match.start()].start
        end = alignment[match.end()].start if match.end() < len(alignment) else None
        words.append(TokenSpan(start, end))
    return words
