import json
import pytest
from spaniq.monitor.baseline_store import BaselineStore, _prompt_hash


@pytest.fixture
def store(tmp_path):
    return BaselineStore(db_path=str(tmp_path / "test.db"))


def test_db_created_on_init(tmp_path):
    db = tmp_path / "init_test.db"
    BaselineStore(db_path=str(db))
    assert db.exists()


def test_create_and_get(store):
    bid = store.create(name="test", prompt="hello?", outputs=["a", "b", "c"])
    b = store.get(bid)
    assert b.id == bid
    assert b.name == "test"
    assert json.loads(b.outputs) == ["a", "b", "c"]
    assert b.n_outputs == 3
    assert b.version == 1


def test_get_by_name(store):
    bid = store.create(name="mybase", prompt="what?", outputs=["x"])
    b = store.get_by_name("mybase")
    assert b.id == bid


def test_get_by_prompt(store):
    store.create(name="p1", prompt="unique prompt text", outputs=["out1"])
    b = store.get_by_prompt("unique prompt text")
    assert b is not None
    assert b.name == "p1"


def test_missing_baseline_returns_none(store):
    result = store.get_by_prompt("nonexistent prompt xyz")
    assert result is None


def test_update_increments_version(store):
    bid = store.create(name="v", prompt="p", outputs=["a"])
    store.update(bid, ["b"])
    b = store.get(bid)
    assert b.version == 2


def test_update_appends_outputs(store):
    bid = store.create(name="app", prompt="p", outputs=["a", "b"])
    store.update(bid, ["c", "d"])
    b = store.get(bid)
    assert json.loads(b.outputs) == ["a", "b", "c", "d"]
    assert b.n_outputs == 4


def test_list_all(store):
    store.create(name="b1", prompt="p1", outputs=["x"])
    store.create(name="b2", prompt="p2", outputs=["y", "z"])
    summaries = store.list_all()
    names = [s.name for s in summaries]
    assert "b1" in names
    assert "b2" in names


def test_delete(store):
    bid = store.create(name="del", prompt="bye", outputs=["a"])
    store.delete(bid)
    with pytest.raises(KeyError):
        store.get(bid)


def test_prompt_hash_deterministic():
    h1 = _prompt_hash("same prompt")
    h2 = _prompt_hash("same prompt")
    assert h1 == h2


def test_prompt_hash_different_for_different_prompts():
    h1 = _prompt_hash("prompt one")
    h2 = _prompt_hash("prompt two")
    assert h1 != h2
