from hashlib import md5
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from openai import OpenAI
from dataclasses import dataclass
import json
@dataclass
class LLMResponse:
    """LLM响应的数据类"""
    content: str
    model: str
    raw_response: Any

class LLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self._initialize_client()
        self.times = 0
        self.__init_cache()
    
    def __init_cache(self):
        cache_file = os.path.join("cache", str(self.__class__.__name__) + "_cache.json")
        os.makedirs("cache", exist_ok=True)
        self.cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                self.cache = json.load(f)
        
    def save_cache(self):
        cache_file = os.path.join("cache", str(self.__class__.__name__) + "_cache.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the specific LLM client"""
        pass
    
    
    def chat(self, 
            prompt: str,
            model: Optional[str] = None,
            max_tokens: int = 1000,
            temperature: float = 0.7
            ) -> LLMResponse:
        """Send a chat request to the LLM"""

        key = md5(prompt.encode('utf-8')).hexdigest()
        if key in self.cache:
            return self.cache[key]
        
        self.times += 1
        print(f"调用次数: {self.times}")
        
        response = self._chat(prompt, model, max_tokens, temperature)
        self.cache[key] = response.content
        self.save_cache()
        return response.content
        
    @abstractmethod
    def _chat(self, 
            prompt: str,
            model: Optional[str] = None,
            max_tokens: int = 1000,
            temperature: float = 0.7
            ) -> LLMResponse:
        pass

class GPTClient(LLMClient):
    """OpenAI GPT客户端实现"""
    
    DEFAULT_MODEL = "gpt-4o-mini"
    
    def _initialize_client(self) -> None:
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _chat(self,
             prompt: str,
             model: Optional[str] = None,
             max_tokens: int = 1000,
             temperature: float = 0.7
             ) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=model or self.DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model or self.DEFAULT_MODEL,
                raw_response=response
            )
        except Exception as e:
            raise Exception(f"GPT API调用错误: {str(e)}")

class QwenClient(LLMClient):
    """Qwen(千问)客户端实现"""
    
    DEFAULT_MODEL = "qwen-turbo"
    
    def _initialize_client(self) -> None:
        # 在这里初始化千问的API客户端
        # 由于千问也支持OpenAI兼容格式，我们使用OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _chat(self,
             prompt: str,
             model: Optional[str] = None,
             max_tokens: int = 1000,
             temperature: float = 0.7
             ) -> LLMResponse:
        super().chat(prompt, model, max_tokens, temperature)
        try:
            response = self.client.chat.completions.create(
                model=model or self.DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model or self.DEFAULT_MODEL,
                raw_response=response
            )
        except Exception as e:
            raise Exception(f"千问 API调用错误: {str(e)}")

class LLMFactory:
    """LLM客户端工厂类"""
    
    @staticmethod
    def create_client(client_type: str, api_key: str, base_url: str) -> LLMClient:
        """Create a specific LLM client
        
        Args:
            client_type: 'gpt' or 'qwen'
            api_key: API key
            base_url: Base URL for the API
        
        Returns:
            LLMClient instance
        """
        if client_type.lower() == 'gpt':
            return GPTClient(api_key=api_key, base_url=base_url)
        elif client_type.lower() == 'qwen':
            return QwenClient(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

# 使用示例
def main():
    # 从环境变量获取API密钥
    api_key = os.getenv('OPENAI_API_KEY', 'sk-11TR1NSdoqpvK10I53E689B8D0584eE5938bE321B0Ca955b')
    
    # 创建GPT客户端
    gpt_client = LLMFactory.create_client(
        'gpt',
        api_key=api_key,
        base_url="https://api2.mygptlife.com/v1"
    )
    
    # # 创建千问客户端
    # qwen_client = LLMFactory.create_client(
    #     'qwen',
    #     api_key=api_key,
    #     base_url="https://your-qianwen-api-endpoint.com/v1"
    # )
    
    # 测试提示词
    prompt = "请用中文解释什么是Python装饰器？"
    
    # 使用GPT进行对话
    try:
        gpt_response = gpt_client.chat(prompt)
        print(f"GPT的回答:\n{gpt_response.content}\n")
    except Exception as e:
        print(f"GPT调用失败: {str(e)}")
    
    # 使用千问进行对话
    # try:
    #     qwen_response = qwen_client.chat(prompt)
    #     print(f"千问的回答:\n{qwen_response.content}")
    # except Exception as e:
    #     print(f"千问调用失败: {str(e)}")

if __name__ == "__main__":
    main()
    