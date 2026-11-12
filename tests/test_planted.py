"""Planted-error corruptions: deterministic, and each one really changes the draft. No API calls."""

from src.multi_agent.planted import (
    add_unsupported_claim, change_number, corrupt_all, swap_entity,
)

DRAFT = "Inception (2010), directed by Christopher Nolan, has a rating of 8.1 and grossed $835,532,764."


def test_swap_entity_changes_director_to_one_not_in_evidence():
    c = swap_entity(DRAFT, "Director: Christopher Nolan")
    assert c and "Christopher Nolan" not in c.draft and c.type == "swap_entity"
    assert c.detail.split(" -> ")[1] not in "Director: Christopher Nolan"


def test_swap_entity_none_when_no_known_director():
    assert swap_entity("Avatar was released in 2009.", "") is None


def test_change_number_alters_first_data_number():
    c = change_number(DRAFT)
    assert c and "2013" in c.draft and "2010" not in c.draft


def test_change_number_skips_list_markers():
    c = change_number("1. Avatar (2009)\n2. Aliens (1986)")
    assert c and c.draft.startswith("1. Avatar (2012)")


def test_change_number_none_when_no_numbers():
    assert change_number("No numbers here.") is None


def test_unsupported_claim_is_appended():
    c = add_unsupported_claim(DRAFT)
    assert c.draft.startswith(DRAFT) and len(c.draft) > len(DRAFT)


def test_corrupt_all_returns_only_applicable_changed_drafts():
    cs = corrupt_all(DRAFT, "Director: Christopher Nolan")
    assert {c.type for c in cs} == {"swap_entity", "change_number", "add_unsupported_claim"}
    assert all(c.draft != DRAFT for c in cs)
    assert [c.type for c in corrupt_all("Plain text.", "")] == ["add_unsupported_claim"]


from src.multi_agent.planted import corrupt_hard, drop_list_item, near_miss_number, swap_years

LIST = "1. Avatar (2009)\n2. Aliens (1986)\n3. The Abyss (1989)\n4. True Lies (1994)"


def test_near_miss_number_is_small():
    assert near_miss_number(DRAFT).draft.startswith("Inception (2011)")
    assert "8.2" in near_miss_number("rated 8.1").draft
    assert "1,030" in near_miss_number("earned 1,000").draft


def test_near_miss_skips_list_markers():
    assert near_miss_number(LIST).draft.startswith("1. Avatar (2010)")


def test_drop_list_item_removes_middle_line_only():
    c = drop_list_item(LIST)
    assert c and "The Abyss" not in c.draft and "Aliens" in c.draft and c.draft.count("\n") == 2


def test_drop_list_item_needs_three_items():
    assert drop_list_item("1. A\n2. B") is None


def test_swap_years_trades_first_two_distinct_years():
    c = swap_years(LIST)
    assert c and c.draft.startswith("1. Avatar (1986)\n2. Aliens (2009)")
    assert swap_years("Released in 2009 and re-released in 2009.") is None


def test_corrupt_hard_only_applicable_changed():
    assert {c.type for c in corrupt_hard(LIST)} == {"near_miss_number", "drop_list_item", "swap_years"}
    assert corrupt_hard("No data.") == []
