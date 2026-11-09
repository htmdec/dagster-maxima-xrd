from __future__ import annotations

from maxima_dagster import sensors


def test_parse_girder_cursor_handles_invalid_json() -> None:
    since, checksums, pending = sensors._parse_girder_cursor("not-json")
    assert since == ""
    assert checksums == {}
    assert pending == {}


def test_parse_girder_cursor_handles_non_dict_json() -> None:
    since, checksums, pending = sensors._parse_girder_cursor('["not", "dict"]')
    assert since == ""
    assert checksums == {}
    assert pending == {}


def test_parse_girder_cursor_filters_blank_checksum_entries() -> None:
    cursor = sensors._serialize_girder_cursor(
        "2026-01-01T00:00:00+00:00",
        {
            "exp_1": "chk_1",
            "": "skip",
            "exp_2": "",
        },
        {"exp_3": "chk_3", "exp_4": ""},
    )

    since, checksums, pending = sensors._parse_girder_cursor(cursor)

    assert since == "2026-01-01T00:00:00+00:00"
    assert checksums == {"exp_1": "chk_1"}
    assert pending == {"exp_3": "chk_3"}


def test_parse_girder_cursor_accepts_legacy_cursor_without_pending() -> None:
    cursor = '{"since": "1970-01-01T00:00:00.000000+00:00", "checksums_by_partition": {"exp_1": "chk_1"}}'

    since, checksums, pending = sensors._parse_girder_cursor(cursor)

    assert since == "1970-01-01T00:00:00.000000+00:00"
    assert checksums == {"exp_1": "chk_1"}
    assert pending == {}