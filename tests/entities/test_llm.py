from sherma.entities.llm import LLM


def test_llm_creation():
    llm = LLM(id="gpt4", version="1.0.0", model_name="gpt-4")
    assert llm.model_name == "gpt-4"
