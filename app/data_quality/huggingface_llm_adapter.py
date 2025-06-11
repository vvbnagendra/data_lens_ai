from pandasai.llm.base import LLM
import requests
import os

class HuggingFaceLLM(LLM):
    def __init__(self, model="deepseek/deepseek-r1-0528", token=None):
        super().__init__()  # ✅ REQUIRED!
        self.model = model
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
        self.api_url = "https://router.huggingface.co/novita/v3/openai/chat/completions"

    def call(self, prompt: str, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        messages = [
            {
                "role": "system",
                "content": "You are a Python data analyst. Only return Python code using a DataFrame called `df`."
            },
            {
                "role": "user",
                "content": str(prompt)  # ✅ Ensure string
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
        }

        print(f"Calling Hugging Face API with payload: {payload}")

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            print(f"Response status code: {response.status_code}")
            result = response.json()
            print(f"Response JSON: {result}")
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error calling Hugging Face API: {e}"

    @property
    def type(self) -> str:
        return "huggingface"  # ✅ REQUIRED by PandasAI
