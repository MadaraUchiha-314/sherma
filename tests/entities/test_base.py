from sherma.entities.base import EntityBase


def test_entity_base_defaults():
    e = EntityBase(id="test")
    assert e.id == "test"
    assert e.version == "*"


def test_entity_base_with_version():
    e = EntityBase(id="test", version="1.0.0")
    assert e.version == "1.0.0"
