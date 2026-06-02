from typing import TypeAlias

import pytest
from transformers import ByT5Tokenizer, RobertaTokenizer

from htrflow.utils import token_alignment


Tokenizer: TypeAlias = ByT5Tokenizer | RobertaTokenizer


templates = [
    "some completely normal words",
    "somé wôrds with chärachters þat míght span several tökens",
    " some words with leading and trailing whitespace ",
    "  ",  # a whitespace-only string
    "aSingleWord",
    "some numbers 12345 and punctuation :,.?;",
    "some {unk_token}words sprinkled {unk_token} with spe{unk_token}cial tokens {unk_token}",
]


@pytest.fixture(scope="module")
def byt5_tokenizer():
    return ByT5Tokenizer()


@pytest.fixture(scope="module")
def roberta_tokenizer():
    yield RobertaTokenizer.from_pretrained("Riksarkivet/trocr-base-handwritten-hist-swe-2")


@pytest.fixture(params=["byt5_tokenizer", "roberta_tokenizer"], scope="module")
def tokenizer(request) -> Tokenizer:
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize("template", templates)
def test_align_chars_to_tokens(tokenizer: Tokenizer, template: str):
    """
    Tests `align_chars_to_tokens` on a bunch of cases.
    """
    input_text = template.format(**tokenizer.special_tokens_map)
    token_ids = tokenizer.encode(input_text)
    alignment = token_alignment.align_chars_to_tokens(token_ids, tokenizer)

    expected_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert len(alignment) == len(expected_text)

    # We expect each TokenSpan to contain its corresponding character
    for i, (span, character) in enumerate(zip(alignment, expected_text)):
        text_chunk = tokenizer.decode(token_ids[span.start : span.end], skip_special_tokens=True)
        assert character in text_chunk


@pytest.mark.parametrize("template", templates)
def test_align_words_to_tokens(tokenizer: Tokenizer, template: str):
    """
    Tests `align_words_to_tokens` on a bunch of cases.
    """
    text = template.format(**tokenizer.special_tokens_map)
    token_ids = tokenizer.encode(text)
    word_spans = token_alignment.align_words_to_tokens(token_ids, tokenizer)

    expected_words = tokenizer.decode(token_ids, skip_special_tokens=True).split()
    assert len(word_spans) == len(expected_words)

    for span, expected_word in zip(word_spans, expected_words):
        # The computed span should correspond to the expected word...
        word = tokenizer.decode(token_ids[span.start : span.end], skip_special_tokens=True)
        assert word.strip() == expected_word

        # ... and should also be the smallest possible span that correspond
        # to the expected word. (i.e., it should not include any leading or
        # trailing special tokens or whitespace)
        shorter_word_1 = tokenizer.decode(token_ids[span.start + 1 : span.end], skip_special_tokens=True)
        assert shorter_word_1.strip() != expected_word

        if span.end is not None and span.end > 0:
            shorter_word_2 = tokenizer.decode(token_ids[span.start : span.end - 1], skip_special_tokens=True)
            assert shorter_word_2.strip() != expected_word
