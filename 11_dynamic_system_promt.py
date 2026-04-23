from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from init_model import deep_model


class UserRole(TypedDict):
    user_role: str
    content: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色动态生成系统提示"""
    user_role = request.runtime.context["user_role"]
    print(f"用户角色: {user_role}")
    base_prompt = "你是个有用的助手."

    if user_role == "专家":
        return f"{base_prompt} 提供详细的专业的技术回答."
    elif user_role == "新手小白":
        return f"{base_prompt} 解释概念，避免使用专业术语。."
    return base_prompt

agent = create_agent(
    model = deep_model,
    middleware = [user_role_prompt],
)

response = agent.invoke({"messages": [{"role": "user", "content": "什么是langchain？"}]},
             context= UserRole(user_role="新手小白", content="你是个专业的助手."))
print(response["messages"][-1].content)
