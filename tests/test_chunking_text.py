from app.chunking import chunk_pages, split_sentences
from app.text import locate_quote, normalize_ws, numbers_in


def test_normalize_ws():
    assert normalize_ws("  a\n\n b\t c ") == "a b c"


def test_locate_quote_across_line_break_returns_original_span():
    text = "Intro. UV damage to tent\nfabrics is excluded after the first 12 months. End."
    span = locate_quote(text, "UV damage to tent fabrics is excluded after the first 12 months.")
    assert span is not None
    assert text[span[0] : span[1]] == "UV damage to tent\nfabrics is excluded after the first 12 months."


def test_locate_quote_is_strict_about_everything_but_whitespace():
    text = "Customers may return unused items within 30 days of delivery."
    assert locate_quote(text, "customers may return unused items") is None  # case
    assert locate_quote(text, "within 31 days") is None
    assert locate_quote(text, "Customers may return unused items within 30 days of delivery") is not None
    assert locate_quote(text, "   ") is None


def test_numbers_in():
    assert numbers_in("Pay 1,500 USD in 30 days at 8.40 each") == ["1,500", "30", "8.40"]


def test_split_sentences_keeps_decimals():
    assert split_sentences("Zone 2 costs 8.40 USD. Zone 3 costs 9.10 USD.") == [
        "Zone 2 costs 8.40 USD.",
        "Zone 3 costs 9.10 USD.",
    ]


def test_chunks_never_cross_pages_and_keep_page_numbers():
    pages = [(1, "First page sentence one. " * 40), (2, "Second page sentence. " * 40)]
    chunks = chunk_pages(pages, max_chars=300)
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        if c.page == 1:
            assert "Second" not in c.text
        else:
            assert "First" not in c.text
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_overlap_keeps_two_sentence_quotes_whole():
    sents = [f"Sentence number {i} has some words in it." for i in range(30)]
    chunks = chunk_pages([(None, " ".join(sents))], max_chars=200)
    for a, b in zip(sents, sents[1:], strict=False):
        assert any(f"{a} {b}" in normalize_ws(c.text) for c in chunks)


def test_long_sentence_is_not_cut():
    long = "word " * 300 + "end."
    chunks = chunk_pages([(None, long)], max_chars=100)
    assert len(chunks) == 1 and chunks[0].text.endswith("end.")


def test_heading_is_glued_to_the_next_text_and_lists_are_units():
    text = "# Title\n\nIntro text here.\n\n## Rates\n\n- Zone 1 costs 5.90 USD.\n- Zone 2 costs 8.40 USD."
    chunks = chunk_pages([(None, text)], max_chars=40)
    assert not any(c.text.strip() in ("# Title", "## Rates") for c in chunks)
    assert any("- Zone 2 costs 8.40 USD." in c.text for c in chunks)


def test_empty_page_gives_no_chunks():
    assert chunk_pages([(1, "   \n  ")]) == []


def test_quote_cannot_start_or_end_inside_a_number_or_word():
    text = "Refunds are accepted within 130 days. Bulk price is 1,500 units at most."
    assert locate_quote(text, "30 days. Bulk") is None
    assert locate_quote(text, "500 units at") is None
    assert locate_quote(text, "efunds are accepted") is None
    assert locate_quote(text, "within 130 days.") is not None
    assert locate_quote(text, "1,500 units") is not None


def test_quote_found_at_a_later_clean_position():
    text = "Item 130 days. Also 30 days applies."
    span = locate_quote(text, "30 days")
    assert span is not None and text[span[0] - 1] == " "


def test_heading_only_page_still_becomes_a_chunk():
    chunks = chunk_pages([(1, "# Supplier Terms"), (2, "Body text here.")])
    assert [c.page for c in chunks] == [1, 2]
