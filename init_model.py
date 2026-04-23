import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, wrap_model_call, ModelResponse
from langchain.chat_models import init_chat_model

load_dotenv()

deep_model = init_chat_model(
    model=os.getenv("DEEPSEEK_MODEL"),
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE"),
)

qianwen_model = init_chat_model(
    model=os.getenv("QWEN_MODEL"),
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_API_BASE"),
)