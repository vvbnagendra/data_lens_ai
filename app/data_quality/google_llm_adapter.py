# google_llm_adapter.py

from pandasai.llm.base import LLM
import google.generativeai as genai
import os
import ssl
import urllib3
import re

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Force Python to use system certs OR disable SSL verification (INSECURE)
ssl._create_default_https_context = ssl._create_unverified_context


class GoogleLLM(LLM):
    def __init__(self, model="gemini-1.5-flash", api_key=None):
        super().__init__()
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required for GoogleLLM.")

        # Configure Gemini API client
        genai.configure(api_key=self.api_key)

    def call(self, prompt: str, **kwargs):
        try:
            model = genai.GenerativeModel(self.model)
            chat = model.start_chat()

            system_prompt = (
                "You are a Python data analyst. "
                "Only return Python code using a DataFrame called df. "
                "Don't explain. Don't output anything other than code."
            )
            final_prompt = f"{system_prompt}\n\nUser request:\n{str(prompt)}"
            print(f"Calling Gemini API with prompt: {final_prompt}")

            response = chat.send_message(final_prompt)
            output = response.text.strip()
            print(f"Raw Response: {output}")

            # Extract code if wrapped in python
            code_match = re.search(r"(?:python)?\n(.*?)```", output, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

            return output
        except Exception as e:
            return f"Error calling Gemini API: {e}"

    @property
    def type(self) -> str:
        return "google"