"""This tests the core part of `text_data`: the `Corpus`.

Because much of the core of this library is written in Rust,
all of the tests for `WordIndex` are in the Rust library.

But this validates the functionality of searching for documents,
adding n-gram indexes, and displaying results.
"""
from faker import Faker
import pytest

from text_data import Corpus, PositionResult


def test_empty_corpus():
    """Initializing a `Corpus` with no documents should work."""
    corpus = Corpus([])
    assert len(corpus) == 0
    assert corpus.vocab == set()
    assert corpus.most_common() == []
    assert corpus.vocab_size == 0
    assert corpus.num_words == 0


def test_basic_initialization():
    """This makes sure that initializing the Corpus works as expected."""
    corpus = Corpus(["I ran to the park with the baseball."])
    assert corpus.most_common(1) == [("the", 2)]
    assert corpus.vocab == {"i", "ran", "to", "the", "park", "with", "baseball"}
    assert corpus.vocab_size == 7
    assert corpus.num_words == 8
    assert len(corpus) == 1
    assert corpus.word_count("the") == 2
    assert corpus.word_frequency("the") == 0.25
    assert corpus.document_count("the") == 1
    assert corpus.document_frequency("the") == 1.0
    assert corpus.term_count("the", 0) == 2


def test_chunking():
    """Tests `text_data.Corpus.chunks`."""
    documents = ["I ran to the park with the baseball."] * 2
    for corpus in Corpus.chunks(documents, chunksize=1):
        assert corpus.most_common(1) == [("the", 2)]
        assert corpus.vocab_size == 7
        assert corpus.num_words == 8
        assert len(corpus) == 1


@pytest.mark.parametrize(
    "sep,prefix,suffix,default,bigram",
    [
        (None, None, None, True, "of the"),
        ("</w><w>", "<w>", "</w>", False, "<w>of</w><w>the</w>"),
        ("</w><w>", "<w>", "</w>", True, "of the"),
    ],
)
def test_ngram_index(sep, prefix, suffix, default, bigram):
    """Tests capabilities adding n-grams (`text_data.Corpus.add_ngram_index`)."""
    corpus = Corpus(["of the best or of the worst"])
    assert 2 not in corpus.ngram_indexes
    corpus.add_ngram_index(2, default, sep, prefix, suffix)
    assert 2 in corpus.ngram_indexes
    assert corpus.ngram_indexes[2].most_common(1) == [(bigram, 2)]


def test_update():
    """This makes sure you can update documents."""
    fake = Faker()
    corpus = Corpus([fake.paragraph() for _ in range(5)])
    assert len(corpus) == 5
    corpus.add_ngram_index(n=2)
    assert len(corpus.ngram_indexes[2]) == 5
    new_docs = [fake.paragraph() for _ in range(5)]
    corpus.update(new_docs)
    assert len(corpus) == 10
    assert len(corpus.ngram_indexes[2]) == 10


@pytest.mark.parametrize(
    "query,output",
    [
        ("'truth is'", {0, 1}),
        ("truth is", {0, 1, 2, 4}),
        ("truth OR is", {0, 1, 2, 3, 4, 5, 6}),
        ("'truth is' AND there", {0}),
        ("truth NOT no", {0, 1, 2, 3}),
        ("truth OR cat is", {0, 1, 2, 3, 4, 6}),
        ("not relevant", set()),
    ],
)
def test_document_search(query, output):
    """This makes sure that searching for documents works as expected."""
    corpus = Corpus(
        [
            "The truth is out there",
            "Truth is, I don't know",
            "Is it truth?",
            "Truth",
            "He is no friend of the truth",
            "is it",
            "the cat is happy",
        ]
    )
    assert corpus.search_documents(query) == output


@pytest.mark.parametrize(
    "query, output",
    [
        (
            "dog",
            {
                PositionResult(0, 1, 1, 4, 7),
                PositionResult(1, 1, 1, 4, 7),
                PositionResult(1, 6, 6, 25, 28),
            },
        ),
        ("'the cat'", {PositionResult(2, 0, 1, 0, 7), PositionResult(0, 4, 5, 15, 22)}),
        ("nonsense", set()),
    ],
)
def test_phrase_search(query, output):
    """Makes sure searching for individual instances of a query works."""
    corpus = Corpus(
        ["The dog ran to the cat", "The dog ran to the other dog", "The cat sat"]
    )
    assert corpus.search_occurrences(query) == output


@pytest.mark.parametrize(
    "query,output",
    [
        ("example", [0, 2]),
        ("search", [1, 2, 0]),
        # combined ranks give extra weight to words with higher TF-IDF scores
        ("example search", [0, 2]),
        # adjust ranks based on number of query matches
        ("example search search", [2, 0]),
        # empty searches should just be empty
        ("nonsense", []),
    ],
)
def test_document_ordering(query, output):
    """Tests that the document ordering of `ranked_search` follows TF-IDF."""
    document_words = [
        ["example"] * 80 + ["search"] * 5,
        ["search"] * 5,
        ["example"] + ["search"] * 5,
    ]
    corpus = Corpus([" ".join(doc) for doc in document_words])
    ranked = corpus.ranked_search(query)
    documents = [doc[0].doc_id for doc in ranked]
    assert documents == output


@pytest.mark.parametrize(
    "query,output",
    [
        # searching for an item that only appears once should only return that result
        ("hat", [[PositionResult(0, 4, 4, 16, 19)]]),
        # searching for an item that appears twice will next be sorted by the length of the phrase
        (
            "'the cat' hat",
            [[PositionResult(0, 0, 1, 0, 7), PositionResult(0, 4, 4, 16, 19)]],
        ),
        # searching for items with equal phrase lengths (defined by the query) will be sorted
        # by the distance between each phrase and the last occurring phrase
        (
            "the hat",
            [
                [
                    PositionResult(0, 4, 4, 16, 19),
                    PositionResult(0, 3, 3, 12, 15),
                    PositionResult(0, 0, 0, 0, 3),
                ]
            ],
        ),
    ],
)
def test_document_positioning(query, output):
    """This makes sure that the occurrences within the ranked search appear in the right order."""
    corpus = Corpus(["The cat and the hat and another cat"])
    ranked = corpus.ranked_search(query)
    assert ranked == output


def test_html_display():
    """This tests `text_data.Corpus.display_search_results`."""
    corpus = Corpus(["The cat ran to the dog", "The dog likes bones"])
    # this should only show one result
    html_display = corpus._show_html_occurrences("the dog", max_results=1)
    assert html_display == (
        "<p><b>Showing Result 0 (Document ID 1)</b></p>"
        "<p style='white-space=pre-wrap;'>"
        "<b>The</b> <b>dog</b> likes bones"
        "</p>"
    )
    next_result = (
        "<p><b>Showing Result 1 (Document ID 0)</b></p>"
        "<p style='white-space=pre-wrap;'>"
        "<b>The</b> cat ran to <b>the</b> <b>dog</b>"
        "</p>"
    )
    # now, return all results
    all_results = corpus._show_html_occurrences("the dog")
    assert all_results == html_display + next_result
    # if you narrow the window size, return only nearby results.
    small_window = corpus._show_html_occurrences("the dog", window_size=2)
    assert small_window == (
        "<p><b>Showing Result 0 (Document ID 1)</b></p>"
        "<p style='white-space=pre-wrap;'>"
        "<b>The</b> <b>dog</b> l<b>&hellip;</b>"
        "</p>"
        "<p><b>Showing Result 1 (Document ID 0)</b></p>"
        "<p style='white-space=pre-wrap;'>"
        "<b>&hellip;</b><b>the</b> <b>dog</b>"
        "</p>"
    )


@pytest.mark.parametrize(
    "query,document_count,occurrence_count,document_freq",
    [
        ("food", 4, 6, 4 / 5),
        ("fight", 3, 3, 3 / 5),
        ("food fight", 2, 4, 2 / 5),
        ("'food fight'", 1, 1, 1 / 5),
    ],
)
def test_search_metrics(query, document_count, occurrence_count, document_freq):
    """Tests how well the search metrics work.

    The metrics are `Corpus.search_document_count`,
    `Corpus.search_document_freq`, and `Corpus.search_occurrence_count`.
    """
    documents = [
        "the food fight",
        "the fight for food",
        "the boxing fight",
        "the food parade",
        "food food food",
    ]
    corpus = Corpus(documents)
    assert corpus.search_document_count(query) == document_count
    assert corpus.search_occurrence_count(query) == occurrence_count
    assert corpus.search_document_freq(query) == document_freq
