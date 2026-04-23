from typing import TypedDict, Annotated, Callable

from langchain_core.tools import tool
from langgraph.graph import add_messages

from init_model import *


@tool
def public_search(query: str) -> str:
    """公开搜索，查询城市天气"""
    print(f"public result for {query} is 25 degrees Celsius")
    return f"public result for {query} is 25 degrees Celsius"


@tool
def private_search(query: str) -> str:
    """私有搜索，查询城市天气"""
    print(f"private result for {query} is 25 degrees Celsius")
    return f"private result for {query} is 25 degrees Celsius"


@tool
def advanced_private_search(query: str) -> str:
    """高级私有搜索， 查询城市天气"""
    print(f"advanced private result for {query} is 25 degrees Celsius")
    return f"advanced private result for {query} is 25 degrees Celsius"

# 1. 定义 State Schema，必须包含你要用的所有字段
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    authenticated: bool  # 👈 声明这个字段

@wrap_model_call
def state_based_search(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
    """Filter tools based on conversation State."""
    # Read from State: check if user has authenticated
    state = request.state
    print(f"当前请求: {request}")
    print(f"完整 state 内容: {state}")  # 👈 加这行

    is_authenticated = state.get("authenticated", False)
    print(f"is_authenticated is {is_authenticated}")
    message_count = len(state["messages"])

    # Only enable sensitive tools after authentication
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # Limit tools early in conversation
        tools = [t for t in request.tools if t.name.startswith("private_")]
        request = request.override(tools=tools)

    return handler(request)


agent = create_agent(
    model=deep_model,
    tools=[public_search, private_search, advanced_private_search],
    middleware=[state_based_search],
    state_schema = AgentState  # 👈 关键：告诉 LangGraph 用这个 Schema
)

response = agent.invoke({"messages": [{"role": "user", "content": "查询北京天气'"}],
                         "authenticated": True},)
print(response["messages"][-1].content)
