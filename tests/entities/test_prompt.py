from sherma.entities.prompt import Prompt


def test_prompt_creation():
    p = Prompt(id="sys", version="1.0.0", instructions="You are helpful.")
    assert p.id == "sys"
    assert p.instructions == "You are helpful."
