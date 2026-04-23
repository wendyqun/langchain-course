import inspect
from dataclasses import dataclass
from typing import TypedDict, Annotated, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import add_messages
from langgraph.store.memory import InMemoryStore
import langgraph.pregel.main as m

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


class Context(TypedDict):
    user_id: str


@wrap_model_call
def store_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Store preferences."""
    user_id = request.runtime.context["user_id"]
    print(request)
    # Read from Store: get user's enabled features
    store = request.runtime.store
    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        # Only include tools that are enabled for this user
        tools = [t for t in request.tools if t.name in enabled_features]
        request = request.override(tools=tools)

    return handler(request)


store = InMemoryStore()

store.put(
    ("features",),  # namespace
    "user_123",  # key (对应 user_id)
    {
        "enabled_tools": ["advanced_private_search"]  # 该用户可用的工具
    }
)

agent = create_agent(
    model=deep_model,
    tools=[public_search, private_search, advanced_private_search],
    middleware=[store_based_tools],
    store=store,
    context_schema=Context  # 👈 关键：告诉 LangGraph 用这个 Schema
)

# print(inspect.getsource(m.Pregel.stream))

# 3. 调用时传入 context
response = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查询北京的天气"}]},
    context=Context(user_id="user_123")  # 👈 直接作为 invoke 的关键字参数

)
print(response["messages"][-1].content)
