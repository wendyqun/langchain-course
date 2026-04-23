# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

print("DEEPSEEK_API_KEY:", os.getenv("DEEPSEEK_API_KEY"))

# Run the agent
response = agent.invoke({"messages": [{"role": "user", "content": "介绍下你自己，回答如何快速找到ai相关的工作"}]})
print(response)
print(response["messages"][-1].content)
