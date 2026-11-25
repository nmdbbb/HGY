import numpy as np
import pandas as pd
import pytest

from truyen_reco.data import preprocess
from truyen_reco.data.ingest import classify_role


def make_interactions(n_users=40, n_items=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        for item in rng.choice(n_items, size=rng.integers(6, 15), replace=False):
            rows.append({
                "user": f"u{user}", "media_id": int(item),
                "status": rng.choice(["COMPLETED", "CURRENT", "PLANNING",
                                      "DROPPED", "PAUSED"]),
                "score": rng.choice([np.nan, 80.0]),
                "progress": 10, "progress_volumes": 0, "repeat": 0,
                "started_at": None, "completed_at": None,
                "created_at": int(rng.integers(1_600_000_000, 1_700_000_000)),
                "updated_at": 1_700_000_000, "recruit_country": "KR",
            })
    return pd.DataFrame(rows)


def test_catchup_lifts_current_on_ongoing_series():
    """Đọc tới đầu sóng một bộ đang ra phải mạnh ngang đọc hết một bộ đã hoàn.

    Nếu không, mọi truyện đang ra bị dìm, và đang ra gần như trùng manhwa.
    """
    frame = pd.DataFrame([
        {"user": "a", "media_id": 1, "status": "CURRENT", "score": None,
         "progress": 200, "created_at": 1, "updated_at": 1},   # bắt kịp
        {"user": "b", "media_id": 1, "status": "CURRENT", "score": None,
         "progress": 4, "created_at": 1, "updated_at": 1},     # mới đọc
        {"user": "c", "media_id": 1, "status": "COMPLETED", "score": None,
         "progress": 200, "created_at": 1, "updated_at": 1},
    ])
    out = preprocess.label(frame).set_index("user")["confidence"]
    assert out["a"] > out["b"] * 2
    assert out["a"] == pytest.approx(out["c"], rel=0.05)


def test_catchup_falls_back_when_progress_missing():
    frame = pd.DataFrame([
        {"user": "a", "media_id": 9, "status": "CURRENT", "score": None,
         "progress": 0, "created_at": 1, "updated_at": 1},
    ])
    out = preprocess.label(frame)
    assert out["confidence"].iloc[0] == preprocess.CONFIDENCE["CURRENT"]


def test_frontier_ignores_hand_typed_outlier():
    """Một người nhập 'chương 9999' không được kéo lệch thang của cả truyện."""
    frame = pd.DataFrame([
        {"user": f"u{i}", "media_id": 1, "status": "CURRENT", "score": None,
         "progress": 100, "created_at": 1, "updated_at": 1} for i in range(20)
    ] + [{"user": "bad", "media_id": 1, "status": "CURRENT", "score": None,
          "progress": 9999, "created_at": 1, "updated_at": 1}])
    frontier = preprocess.chapter_frontier(frame)
    assert frontier[1] < 500


@pytest.mark.parametrize("role,expected", [
    ("Art (chs 1-179)", ("art",)),
    ("Story (chs 93-201)", ("story",)),
    ("Story & Art", ("story", "art")),
    ("Original Story", ("story",)),
    ("Translator (Polish)", ()),
    ("Lettering (English)", ()),
    ("", ()),
    (None, ()),
])
def test_classify_role(role, expected):
    assert classify_role(role) == expected


def test_planning_is_not_a_positive_signal():
    frame = make_interactions()
    labelled = preprocess.label(frame)
    planning = labelled[labelled["status"] == "PLANNING"]
    assert (planning["signal"] == "wishlist").all()
    assert "positive" not in set(planning["signal"])


def test_dropped_is_negative_not_positive():
    labelled = preprocess.label(make_interactions())
    dropped = labelled[labelled["status"] == "DROPPED"]
    assert (dropped["signal"] == "negative").all()


def test_core_filter_enforces_minimums():
    frame = preprocess.label(make_interactions())
    positives = frame[frame["signal"].isin({"positive", "weak"})]
    core = preprocess.filter_core(positives, min_user=4, min_item=4)
    assert core["user"].value_counts().min() >= 4
    assert core["media_id"].value_counts().min() >= 4


def test_temporal_split_has_no_future_leak():
    frame = preprocess.label(make_interactions())
    train, test, cutoff = preprocess.temporal_split(frame, test_fraction=0.2)
    assert train["event_time"].max() <= cutoff
    assert test["event_time"].min() > cutoff


def test_holdout_users_are_fully_removed_from_train():
    frame = preprocess.label(make_interactions())
    kept, held = preprocess.holdout_users(frame, fraction=0.2, min_items=3)
    assert len(held) > 0
    assert set(kept["user"]) & set(held["user"]) == set()


def test_cold_users_never_appear_in_train():
    result = preprocess.run(make_interactions(n_users=60), min_user=3, min_item=3)
    cold = set(result["cold_users"]["user"])
    assert cold
    assert cold & set(result["train"]["user"]) == set()


def test_dedupe_merges_repeats_keeps_latest_and_drops_hoarders():
    from truyen_reco.data.ingest import dedupe

    rows = [
        {"user": "a", "media_id": 1, "progress": 10, "updated_at": 1},
        {"user": "a", "media_id": 1, "progress": 20, "updated_at": 2},
        {"user": "a", "media_id": 2, "progress": 5, "updated_at": 1},
    ] + [{"user": "bot", "media_id": i, "progress": 1, "updated_at": 1}
         for i in range(50)]
    out = dedupe(pd.DataFrame(rows), max_items_per_user=10)
    assert set(out["user"]) == {"a"}
    assert len(out) == 2
    assert out[out["media_id"] == 1]["progress"].iloc[0] == 20
