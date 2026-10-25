
from promptshield.model import OllamaClient


def test_ollama_returns_text():
    client = OllamaClient()
    result = client.generate("Reply with exactly: PROMPTSHIELD_OK")

    assert isinstance(result, str)
    assert result.strip()
    print("Model response:", result)


if __name__ == "__main__":
    test_ollama_returns_text()
    print("PASS: Ollama client works")

