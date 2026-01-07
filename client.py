from hashlib import md5
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from dataclasses import dataclass
import json
from dotenv import load_dotenv
@dataclass
class LLMResponse:
    content: str
    model: str
    raw_response: Any
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class LLMClient(ABC):
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self._initialize_client()
        self.times = 0
        self.cache = {}
        self.prompt_token_usage = 0
        self.completion_token_usage = 0
        self.total_token_usage = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        
    
    def load_cache(self, model):
        if len(self.cache) > 0:
            return
        cache_file = os.path.join("cache", str(self.__class__.__name__) + "_" + model + "_cache.json")
        os.makedirs("cache", exist_ok=True)
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                self.cache = json.load(f)
    def save_cache(self, model):
        cache_file = os.path.join("cache", str(self.__class__.__name__) + "_" + model + "_cache.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the specific LLM client"""
        pass
    
    
    def chat(self, 
            prompt: str,
            model: Optional[str] = None,
            temperature: float = 0.7
            ) -> LLMResponse:
        """Send a chat request to the LLM"""
        model = model or self.model
        key = md5(prompt.encode('utf-8')).hexdigest()
        self.load_cache(model)
        if key in self.cache:
            self._reset_last_usage()
            cache = self.cache[key]
            if isinstance(cache, str):
                return cache
            return cache['content']
        
        self.times += 1
        print(f"API calling times: {self.times}")
        
        response = self._chat(prompt, model, temperature)
        self._record_token_usage(response)
        self.cache[key] = {
            'content': response.content,
            'prompt_tokens': response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens
        }
        self.save_cache(model)
        return response.content
        
    @abstractmethod
    def _chat(self, 
            prompt: str,
            model: Optional[str] = None,
            temperature: float = 0.7
            ) -> LLMResponse:
        pass
    
    def _reset_last_usage(self) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0

    def _record_token_usage(self, response: LLMResponse) -> None:
        prompt_tokens = response.prompt_tokens or 0
        completion_tokens = response.completion_tokens or 0
        total_tokens = response.total_tokens or (prompt_tokens + completion_tokens)
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens
        self.last_total_tokens = total_tokens
        self.prompt_token_usage += prompt_tokens
        self.completion_token_usage += completion_tokens
        self.total_token_usage += total_tokens

    def _extract_usage_from_response(self, raw_response: Any) -> Tuple[int, int, int]:
        usage = getattr(raw_response, "usage", None)
        if usage is None and isinstance(raw_response, dict):
            usage = usage or raw_response.get("usage")

        def _value(container: Any, name: str) -> int:
            if container is None:
                return 0
            result = None
            if isinstance(container, dict):
                result = container.get(name)
            else:
                result = getattr(container, name, None)
            if result is None:
                return 0
            try:
                return int(result)
            except (TypeError, ValueError):
                return 0

        prompt_tokens = _value(usage, "prompt_tokens")
        completion_tokens = _value(usage, "completion_tokens")
        total_tokens = _value(usage, "total_tokens")
        if prompt_tokens == 0:
            prompt_tokens = _value(usage, "input_tokens")
        if completion_tokens == 0:
            completion_tokens = _value(usage, "output_tokens")
        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens
        return prompt_tokens, completion_tokens, total_tokens
class AntClient(LLMClient):
    
    def _initialize_client(self) -> None:
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _chat(self,
             prompt: str,
             model: Optional[str] = None,
             temperature: float = 0.7
             ) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            prompt_tokens, completion_tokens, total_tokens = self._extract_usage_from_response(response)
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                raw_response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:
            raise Exception(f"ANT API calling error: {str(e)}")
        
class GPTClient(LLMClient):
    
    def _initialize_client(self) -> None:
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _chat(self,
             prompt: str,
             model: Optional[str] = None,
             temperature: float = 0.7
             ) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            prompt_tokens, completion_tokens, total_tokens = self._extract_usage_from_response(response)
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model or self.DEFAULT_MODEL,
                raw_response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:
            raise Exception(f"ANT API calling error: {str(e)}")

class QwenClient(LLMClient):
    def _initialize_client(self) -> None:
        import dashscope
        
        if dashscope is None:
            raise ImportError(
                "dashscope package is required for QwenClient. "
                "Install it with `pip install dashscope`."
            )
        dashscope.api_key = self.api_key
    
    def _chat(self,
             prompt: str,
             model: Optional[str] = None,
             temperature: float = 0.7
             ) -> LLMResponse:
        from dashscope import Generation
        try:
            response = Generation.call(
                model=model,
                prompt=prompt,
                temperature=temperature,
                result_format='text'
            )
            content = response.output.text
            prompt_tokens, completion_tokens, total_tokens = self._extract_usage_from_response(response)
            return LLMResponse(
                content=content,
                model=model,
                raw_response=response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:
            raise Exception(f"Qwen API calling error: {str(e)}")

class LLMFactory:
    
    @staticmethod
    def create_client(client_type: str = None, api_key: str = None, base_url: str = None, model: str = None) -> LLMClient:
        """Create a specific LLM client
        
        Args:
            client_type: 'gpt' or 'qwen'
            api_key: API key
            base_url: Base URL for the API
            model: Model name
        
        Returns:
            LLMClient instance
        """
        load_dotenv()
        if client_type is None:
            client_type = os.getenv('CLIENT_TYPE')
        if client_type.lower() == 'gpt':
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            base_url = base_url or os.getenv('OPENAI_BASE_URL')
            model = os.getenv('OPENAI_MODEL')
            return GPTClient(api_key=api_key, base_url=base_url, model=model)
        elif client_type.lower() == 'qwen':
            api_key = api_key or os.getenv('QWEN_API_KEY')  
            model = os.getenv('QWEN_MODEL')
            return QwenClient(api_key=api_key, base_url=base_url, model=model)
        elif client_type.lower() == "ant":
            api_key = api_key or os.getenv('ANT_API_KEY')
            base_url = base_url or os.getenv('ANT_BASE_URL')
            model = os.getenv('ANT_MODEL')
            return AntClient(api_key=api_key, base_url=base_url, model=model)
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

def main():
    api_key = os.getenv('OPENAI_API_KEY', 'sk-11TR1NSdoqpvK10I53E689B8D0584eE5938bE321B0Ca955b')
    
    gpt_client = LLMFactory.create_client(
        'gpt',
        api_key=api_key,
        base_url="https://api2.mygptlife.com/v1"
    )
    
    # qwen_client = LLMFactory.create_client(
    #     'qwen',
    #     api_key=api_key,
    #     base_url="https://your-qianwen-api-endpoint.com/v1"
    # )
    
    prompt = "what is the decorator in python, use chinese "
    
    try:
        gpt_response = gpt_client.chat(prompt)
        print(f"GPT answer:\n{gpt_response.content}\n")
    except Exception as e:
        print(f"GPT error: {str(e)}")

if __name__ == "__main__":
    main()
    
