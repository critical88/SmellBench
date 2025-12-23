from client import LLMFactory
from dotenv import load_dotenv
import os
load_dotenv()
clientType = "qwen"
client = None
if clientType == "gpt":
    client = LLMFactory.create_client("gpt")
elif clientType == "qwen":
    client = LLMFactory.create_client("qwen")

response = client.chat("Hello, world!")
print(response)