import os
from langchain.tools import tool

from dotenv import load_dotenv
from langchain.agents import create_agent
from init_model import deep_model, qianwen_model

load_dotenv()


@tool
def query_weather(city: str) -> str:
    """根据城市名称查询天气"""
    print(city)
    return f"the weather report for {city} is rainy"


@tool
def search(query: str) -> str:
    """Search the internet for the given query."""
    print(query)
    return f"result for {query}..."


agent  = create_agent(
    model=deep_model,
    tools=[query_weather, search],
)

response = agent.invoke({"messages": [{"role": "user", "content": "北京天气"}]})
print(response["messages"][-1].content)
