"""
context 一般代表不可变配置，如用户角色、用户身份等，根据不同的角色个性化回答
"""

from typing import TypedDict, Callable
from langchain_core.tools import tool
from init_model import *


@tool
def read_data() -> str:
    """读取数据"""
    print("call read_data()")
    return "read data"

@tool
def write_data(data: str) -> str:
    """写入数据"""
    print(f"call write_data({data})")
    return f"write data: {data}"

@tool
def delete_data() -> str:
    """删除数据"""
    print("call delete_data()")
    return "delete data"

class Context(TypedDict):
    user_role: str


@wrap_model_call
def store_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Store preferences."""
    user_role = request.runtime.context["user_role"]
    print(f"user_role: {user_role}")
    print(f"modelRequest: {request}")
    if user_role == "admin":
        # Admins get all tools
        pass
    elif user_role == "editor":
        # Editors get all tools except delete_data
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        # Viewers get read-only tools
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model=deep_model,
    tools=[delete_data, write_data, read_data],
    middleware=[store_based_tools],
    context_schema=Context  # 👈 关键：告诉 LangGraph 用这个 Schema
)

# 3. 调用时传入 context
response = agent.invoke(
    {"messages": [{"role": "user", "content": "删除数据"}]},
    # user_role 为admin时成功调用delete_data工具，为editor时不能调用delete_data工具
    context=Context(user_role="editor")  # 👈 直接作为 invoke 的关键字参数

)
print(response["messages"][-1].content)
