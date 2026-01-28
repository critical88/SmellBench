from client import LLMFactory
from dotenv import load_dotenv
import os
load_dotenv()
clientType = "ant"
client = LLMFactory.create_client(clientType)

response = client.chat("Hello, world!1")
print(response)