"""
state中存储的信息可以被视为agent的短期记忆
通过middleware可以在每次调用时动态调整system prompt
"""
from typing import Any, Callable
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools import tool
from langgraph.graph import state

from init_model import deep_model


# 1. 定义扩展 State（必须继承 AgentState）
class UserPrefState(AgentState):
    user_preferences: dict  # 新增字段

# 2. 定义 Middleware，声明它使用哪个 State
class UserPrefMiddleware(AgentMiddleware):
    state_schema = UserPrefState  # 关键：把 State 绑定在 Middleware 上

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        prefs = request.state.get("user_preferences", {})
        style = prefs.get("style", "normal")
        print(style)
        if style == "technical":
            prompt = "你是一个技术专家，请使用专业术语回答。回答时告诉用户你的角色是技术专家。"
        elif style == "simple":
            prompt = "你是一个老师，请用通俗易懂的语言回答。回答时告诉用户你的角色是老师。"
        else:
            prompt = None

        if prompt:
            # ✅ 正确方式：通过 override 修改本次请求的 system prompt
            updated_request = request.override(system_prompt=prompt)
            print(updated_request)
            return handler(updated_request)

        return handler(request)

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果：{query}"

# 3. 创建 Agent，挂载 Middleware（不需要再传 state_schema）
agent = create_agent(
    model = deep_model,
    tools=[search],
    middleware=[UserPrefMiddleware()],
)

# 4. 调用时在 input 里传入自定义 state 字段
result = agent.invoke({
    "messages": [{"role": "user", "content": "解释一下什么是向量数据库"}],
    "user_preferences": {"style": "technical"},  # 注入自定义 state
})
print(result["messages"][-1].content)
