import requests
from pandasai.llm.base import LLM
import re # Import regex module

# -------------------------------
# Custom LLM Adapter for Ollama
# -------------------------------
class OllamaLLM(LLM):
    def __init__(self, model="deepseek-r1:1.5b", base_url="http://localhost:11434", **kwargs):
        """
        Initializes the OllamaLLM.

        Args:
            model (str): The name of the Ollama model to use (e.g., "deepseek-r1:1.5b").
            base_url (str): The base URL of the Ollama API (e.g., "http://localhost:11434").
            **kwargs: Arbitrary keyword arguments passed to the base LLM class.
        """
        super().__init__(**kwargs) # Correctly pass kwargs to the superclass
        self.model = model
        self.base_url = base_url

    def call(self, instruction, context=None, **kwargs):
        """
        Calls the Ollama API to generate a response based on the instruction.

        Args:
            instruction (object): The prompt instruction object (e.g., GeneratePythonCodePrompt).
            context (dict, optional): Additional context for the LLM call. Defaults to None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The generated Python code or an error message.
        """
        # Define the system prompt for the data analyst role
        system_prompt = (
            "You are a Python data analyst. "
            "Only return Python code using a DataFrame called df. "
            "Don't explain. Don't output anything other than code. "
            "Do not include comments or explanations in the code unless explicitly asked."
        )
        
        # Combine system prompt with the user instruction
        # Convert instruction object to string using str()
        final_prompt = f"{system_prompt}\n\nUser request:\n{str(instruction)}"
        print(f"Calling Ollama LLM with prompt: {final_prompt}")
        
        try:
            # Send prompt to Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": final_prompt, # Use the combined prompt
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.0), # Allow temperature to be passed
                        "top_p": kwargs.get("top_p", 0.9), # Allow top_p
                        "num_ctx": kwargs.get("num_ctx", 4096) # Context window
                    }
                },
                timeout=kwargs.get("request_timeout", 600) # Add a timeout
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            # Debug raw output
            print("Raw Ollama response:")
            print(response.text)

            # Safely parse JSON
            json_data = response.json()
            output = json_data.get("response", "").strip() # Use .get for safety
            print("Parsed Ollama response:", output)

            # --- MODIFIED CODE EXTRACTION ---
            # Pattern to find code blocks: ```python (or other language) followed by code, then ```
            code_match = re.search(r"```(?:python)?\s*\n(.*?)```", output, re.DOTALL)
            if code_match:
                extracted_code = code_match.group(1).strip()
                # Remove any leading/trailing blank lines from the extracted code
                return "\n".join([line for line in extracted_code.splitlines() if line.strip()])
            # --- END MODIFIED CODE EXTRACTION ---
            
            # If no code block is found, return the raw output (might be an explanation or direct code)
            return output
        
        except requests.exceptions.RequestException as e:
            # Catch all requests-related exceptions (connection errors, timeouts, HTTP errors)
            print(f"❌ Error calling Ollama API: {e}")
            return f"Error calling Ollama API: {e}"
        except Exception as e:
            # Catch any other unexpected errors
            print(f"❌ An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"

    @property
    def type(self):
        """Returns the type of the LLM."""
        return "ollama"