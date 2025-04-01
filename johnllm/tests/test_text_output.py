import pytest
from johnllm.src.core import LLMModel
from dotenv import load_dotenv

class TestTextOutput:
    @classmethod
    def setup_class(cls):
        load_dotenv()

    @pytest.mark.parametrize("model_name", ["gpt-4o", "claude", "deepseek/deepseek-chat"])
    def test_text_output(self, model_name):
        message = [{
            "role": "user",
            "content": "What is the capital of France?"
        }]

        model = LLMModel()
        res = model.invoke(message, model_name=model_name, response_format=None)

        assert isinstance(res, str)
        assert len(res) > 0
