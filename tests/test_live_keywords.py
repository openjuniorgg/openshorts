"""Keyword gate (English and Portuguese) and the GTA VI-only stream gate."""
import pytest

from live_keywords import find_hits, is_candidate, is_gta6_stream, normalize


def test_normalize_drops_accents_and_case_and_curly_apostrophes():
    assert normalize("VITÓRIA") == "vitoria"
    assert normalize("Let’s GO") == "let's go"
    assert normalize("  não   é  possível ") == "nao e possivel"


@pytest.mark.parametrize("text, category", [
    ("hahahahaha", "laughter"),
    ("jajajaja", "laughter"),
    ("kkkkkkkk", "laughter"),
    ("rsrsrsrs", "laughter"),
    ("morri de rir", "laughter"),
    ("no way", "reaction"),
    ("não acredito", "reaction"),
    ("que loucura", "reaction"),
    ("ai meu deus", "scream"),
    ("ahhhhh", "scream"),
    ("clip that", "clip_request"),
    ("clipar isso", "clip_request"),
    ("headshot", "game_event"),
    ("a polícia", "game_event"),
    ("vitória", "game_event"),
    ("vice city", "game_event"),
])
def test_each_category_matches_english_and_portuguese(text, category):
    assert category in find_hits(text)


def test_no_match_on_plain_talk():
    assert find_hits("so we are going to the store now") == {}


def test_strong_categories_are_enough_alone():
    assert is_candidate(find_hits("clipa isso")) is True
    assert is_candidate(find_hits("ahhhhh")) is True


def test_weak_categories_need_two_hits():
    assert is_candidate(find_hits("insane")) is False
    assert is_candidate(find_hits("insane police")) is True


def test_hits_have_no_numeric_score():
    hits = find_hits("kkkk que isso, the cops!")
    assert all(isinstance(words, list) for words in hits.values())


@pytest.mark.parametrize("title, category", [
    ("GTA 6 day one", "Grand Theft Auto VI"),
    ("first hours in GTA VI", "Just Chatting"),
    ("GTA6 online grind", ""),
    ("GTA 6 trailer reaction", "Grand Theft Auto VI"),
])
def test_gta6_streams_pass(title, category):
    assert is_gta6_stream(title, category) is True


@pytest.mark.parametrize("title, category", [
    ("GTA V roleplay", "Grand Theft Auto V"),
    ("GTA Online heists", "Grand Theft Auto Online"),
    ("Red Dead 2 chill", "Red Dead Redemption 2"),
    ("gta 5 reacting to gta 6", ""),
    ("GTA 6 trailer", "Grand Theft Auto V"),
    ("just chatting", "Just Chatting"),
    ("vice city fun", "Just Chatting"),
    ("", ""),
])
def test_everything_else_is_refused(title, category):
    assert is_gta6_stream(title, category) is False
