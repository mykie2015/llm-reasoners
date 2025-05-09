import os
import openai
import numpy as np
from typing import Optional, Union, Literal
import time

from reasoners.base import LanguageModel, GenerateOutput
from openai import OpenAI

PROMPT_TEMPLATE_ANSWER = 'Your response need to be ended with "So the answer is"\n\n'
PROMPT_TEMPLATE_CONTINUE = "Please continue to answer the last question, following the format of previous examples. Don't say any other words.\n\n"


class OpenAIModel(LanguageModel):
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature=0.0,
        additional_prompt=None,
        backend: Literal["openai", "sglang"] = "openai",
        is_instruct_model: bool = False,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.backend = backend
        self.additional_prompt = additional_prompt
        self.is_instruct_model = is_instruct_model
        self.api_key = api_key
        self.api_base = api_base
        self.__init_client__()

    def __init_client__(self):
        api_key_to_use = self.api_key
        base_url_to_use = self.api_base

        if self.backend == "openai":
            # Key: Use parameter first, then ENV
            if api_key_to_use is None:
                api_key_to_use = os.getenv("OPENAI_API_KEY", None)
            if api_key_to_use is None:
                 raise ValueError("OpenAI API key must be provided either via the 'api_key' parameter or the 'OPENAI_API_KEY' environment variable.")

            # Base URL: Use parameter first, then OPENAI_API_BASE ENV, then default
            if base_url_to_use is None:
                 base_url_to_use = os.getenv("OPENAI_API_BASE", None) # Check OPENAI_API_BASE

        elif self.backend == "sglang":
            # Key: Use parameter first, then ENV (assuming OPENAI_API_KEY might be used)
            if api_key_to_use is None:
                 api_key_to_use = os.getenv("OPENAI_API_KEY", None) # Or a specific SGLANG key env var if available

            # Base URL: Use parameter first, then OPENAI_API_BASE ENV
            if base_url_to_use is None:
                base_url_to_use = os.getenv("OPENAI_API_BASE", None) # Use OPENAI_API_BASE for sglang too
            if base_url_to_use is None:
                 # Update error message to reflect using OPENAI_API_BASE
                 raise ValueError("API base URL must be provided either via the 'api_base' parameter or the 'OPENAI_API_BASE' environment variable when using the 'sglang' backend.")
            # Note: SGLang might not always require an API key depending on the endpoint setup

        else:
            raise ValueError(f"Invalid backend: {self.backend}")

        # Instantiate the client with the determined key and base URL
        # If base_url_to_use is None here for the 'openai' backend, the client will use its default.
        self.client = OpenAI(
            api_key=api_key_to_use,
            base_url=base_url_to_use,
        )

    def generate(
        self,
        prompt: Optional[Union[str, list[str]]],
        max_tokens: int = None,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        rate_limit_per_min: Optional[int] = 20,
        stop: Optional[str] = None,
        logprobs: Optional[int] = None,
        temperature=None,
        additional_prompt=None,
        retry=64,
        **kwargs,
    ) -> GenerateOutput:

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        logprobs = 0 if logprobs is None else logprobs

        if isinstance(prompt, list):
            assert len(prompt) == 1  # @zj: why can't we pass a list of prompts?
            prompt = prompt[0]
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        is_instruct_model = self.is_instruct_model
        if not is_instruct_model:
            # Recheck if the model is an instruct model with model name
            model_name = self.model.lower()
            if (
                ("gpt-3.5" in model_name)
                or ("gpt-4" in model_name)
                or ("instruct" in model_name)
            ):
                is_instruct_model = True

        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                ### GPT 3.5 and higher use a different API
                if is_instruct_model:
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                    )
                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=None,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=0,
                        **kwargs,
                    )
                    return GenerateOutput(
                        text=[choice["text"] for choice in response.choices],
                        log_prob=[choice["logprobs"] for choice in response["choices"]],
                    )

            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        # after 64 tries, still no luck
        raise RuntimeError(
            "GPTCompletionModel failed to generate output, even after 64 tries"
        )

    def get_next_token_logits(
        self,
        prompt: Union[str, list[str]],
        candidates: Union[list[str], list[list[str]]],
        **kwargs,
    ) -> list[np.ndarray]:
        raise NotImplementedError(
            "GPTCompletionModel does not support get_next_token_logits"
        )

    def get_loglikelihood(
        self, prefix: str, contents: list[str], **kwargs
    ) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        # For OpenAI we can't directly get loglikelihoods, so we estimate them
        # This is a very simple implementation - in production you might want to 
        # use a more sophisticated approach
        results = []
        
        for content in contents:
            # Check that the content starts with the prefix
            if not content.startswith(prefix):
                raise ValueError(f"Content must start with prefix, but got prefix='{prefix}' and content='{content}'")
            
            # For OpenAI models, we can only roughly estimate loglikelihoods
            # Here we use a simple heuristic based on content length
            # Higher values = more likely in our simplified model
            suffix = content[len(prefix):]
            score = -0.1 * len(suffix)  # Simple placeholder - longer suffixes are less likely
            results.append(score)
        
        return np.array(results, dtype=np.float32)


if __name__ == "__main__":
    model = OpenAIModel(model="gpt-3.5-turbo")
    print("-------OpenAI client-------")
    print(model.generate(["How to go to Shanghai from Beijing?"]))
    print("-------SGLang client-------")
    model = OpenAIModel(
        model="meta-llama/Llama-3.1-8B-Instruct",
        backend="sglang",
        is_instruct_model=True,
    )
    print(model.generate(["How to go to Shanghai from Beijing?"]))
