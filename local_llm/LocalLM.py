import json
import time

import torch
import ollama
from openai import OpenAI

class LocalLM:

    def __init__(self, model):
        # Initialize the Ollama client
        # self.client = ollama.Client()
        self.client = OpenAI()
        self.model = model

    # def get_llm_response(self, prompt):
    #
    #     # Send the query to the model
    #     response = self.client.generate(model=self.model, prompt=prompt)
    #     return response.response

    def preprocess_and_parse_json(self, response):
        # Remove any leading/trailing whitespace and newlines
        if response.startswith('```json') and response.endswith('```'):
            cleaned_response = response[len('```json'):-len('```')].strip()

        # Parse the cleaned response into a JSON object
        try:
            json_object = json.loads(cleaned_response)
            return json_object
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return None

    def get_llm_response(self, prompt, mode, max_retries=10):
        """
        Send the prompt to the LLM and get back the response.
        Includes handling for GPU memory issues by clearing cache and waiting before retry.
        """
        for attempt in range(max_retries):
            try:
                # Try generating the response
                # response = self.client.generate(model=self.model, prompt=prompt)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that replies in JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=700,
                    n=1
                )
            except Exception as e:
                # This catches errors like the connection being forcibly closed
                print(f"Error on attempt {attempt + 1}: {e}.")
                try:
                    # Clear GPU cache if you're using PyTorch; this may help free up memory
                    torch.cuda.empty_cache()
                    print("Cleared GPU cache.")
                except Exception as cache_err:
                    print("Failed to clear GPU cache:", cache_err)
                # Wait a bit before retrying to allow memory to recover
                time.sleep(2)
                continue

            try:
                tokens = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }

                try:
                    output = self.preprocess_and_parse_json(response.response)
                    if output is None:
                        continue

                    if mode == "rating":
                        # Check if all keys and values are integers (or convertible to integers)
                        all_int = True
                        for k, v in output.items():
                            try:
                                int(k)
                                int(v)
                            except ValueError:
                                all_int = False
                                break
                        if all_int:
                            return output, tokens
                        else:
                            print(f"Keys and values are not integers on attempt {attempt + 1}. Retrying...")
                            continue  # Continue to next attempt
                    else:
                        print(f"Invalid mode: {mode}")
                        return None, tokens

                except json.JSONDecodeError:
                    print(f"Invalid JSON from LLM on attempt {attempt + 1}. Retrying...")
            except Exception as parse_error:
                print("Error processing output:", parse_error)

        print("Max retries exceeded. Returning empty response.")
        return [], {}
