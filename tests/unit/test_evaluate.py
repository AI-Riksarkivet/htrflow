import random

from htrflow import evaluate


def test_evaluate():

    gt_directory = "tests/unit/data/evaluation_test_data/dummy-gt"
    candidate_directory = "tests/unit/data/evaluation_test_data/dummy-htr"
    df = evaluate.evaluate(gt_directory, candidate_directory)
    df.to_csv("test.csv")

    assert df["dummy-htr", "cer"]["equal.xml"] == 0
    assert df["dummy-htr", "wer"]["equal.xml"] == 0
    assert df["dummy-htr", "bow_extras"]["line_level_gt_word_level_htr.xml"] == 0


def test_bow_order_invariance():

    bow = evaluate.BagOfWords()
    words = "apple banana cucumber daikon".split()

    for _ in range(3):
        random.shuffle(words)
        line0 = " ".join(words)

        random.shuffle(words)
        line1 = " ".join(words)

        result = bow.compute(line0, line1)
        assert result["bow_hits"] == 1
        assert result["bow_extras"] == 0


def test_bow_empty_prediction():
    bow = evaluate.BagOfWords()
    line0 = "apple"
    line1 = ""
    result = bow.compute(line0, line1)
    assert result["bow_hits"] == 0, "an empty line should not result in any bag-of-word hits"
    assert result["bow_extras"] == -1, "an empty line should not result in any bag-of-word extras"


def test_bow_empty_ground_truth():
    bow = evaluate.BagOfWords()

    line0 = ""
    line1 = "apple"
    result = bow.compute(line0, line1)
    assert result["bow_hits"] == -1
    assert result["bow_extras"] == len(line1.split())


def test_ratio_addition():
    a = evaluate.Ratio(0, 1)
    b = evaluate.Ratio(1, 9)
    c = evaluate.Ratio(1, 10)
    assert a + b == c
    assert sum((a, b)) == c


def test_ratio_to_float():

    assert float(evaluate.Ratio(0, 0)) == -1
    assert float(evaluate.Ratio(0, 1)) == 0
    assert float(evaluate.Ratio(4, 16)) == 0.25


def test_ratio_add_zero():
    a = evaluate.Ratio(1, 2)
    assert a == a + 0
    assert a == 0 + a


def test_ratio_gt():
    a = evaluate.Ratio(1, 4)
    b = evaluate.Ratio(1, 3)
    assert b > a
    assert b > 0.1

def test_ratio_lt():
    a = evaluate.Ratio(1, 1)
    b = evaluate.Ratio(1, 3)
    assert b < a
    assert b < 0.5


def test_ratio_eq():
    a = evaluate.Ratio(1, 1)
    b = evaluate.Ratio(100, 100)

    assert a == b
    assert 1 == a
    assert a == 1


def test_cer():
    cer = evaluate.CER()
    text0 = "abcdefghijklmnopqrstuvxyzåäö"
    for i in range(len(text0)):
        text1 = "😎" * i + text0[i:]
        result = cer.compute(text0, text1)
        assert result["cer"] == i / len(text0)


def test_cer_empty():
    cer = evaluate.CER()
    something = "lorem ipsum"
    empty = ""

    result = cer.compute(something, empty)
    assert result["cer"] == 1.0

    result = cer.compute(empty, something)
    assert result["cer"] == 1.0


def test_wer():
    wer = evaluate.WER()
    words = list("abcdefghijklmnopqrstuvxyzåäö")
    for i in range(len(words)):
        text0 = " ".join(words)
        text1 = " ".join(["😎"] * i + words[i:])
        result = wer.compute(text0, text1)
        assert result["wer"] == i / len(words)


def test_wer_empty():
    wer = evaluate.WER()
    something = "lorem ipsum"
    empty = ""

    result = wer.compute(something, empty)
    assert result["wer"] == 1.0

    result = wer.compute(empty, something)
    assert result["wer"] == 1.0
