
"""Ollama model client for PromptShield."""

import requests


class OllamaClient:
    """Send prompts to a locally running Ollama model."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
    ) -> None:
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/generate"
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Generate a response for a single prompt."""
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]

