# 动态注册工具
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from init_model import deep_model


@tool
def calculate_tip(total_amount: float, tip_percentage: float = 20) -> str:
    """计算账单的小费金额"""
    print(f"计算{total_amount}元的小费，小费比例为{tip_percentage}%")
    tip_amount = total_amount * (tip_percentage / 100)
    total_amount = total_amount + tip_amount
    return f"Tip amount: {tip_amount}, Total amount: {total_amount}"


@tool
def query_weather(city: str) -> str:
    """根据城市名称查询天气"""
    print(f"查询城市{city}的天气")
    return f"the weather report for {city} is rainy"

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that registers and handles dynamic tools."""
    """
    ModelRequest — 拦截模型调用
    用于 wrap_model_call，在 LLM 被调用之前/之后 介入。
    """
    def wrap_model_call(self, request: ModelRequest, handler):
        # Add dynamic tool to the request
        # This could be loaded from an MCP server, database, etc.
        print("调用wrap_model_call动态注册工具calculate_tip")
        updated = request.override(tools=[*request.tools, calculate_tip])
        return handler(updated)
    """
    ToolCallRequest — 拦截工具调用
    用于 wrap_tool_call，在每次工具被执行时介入。
    这个方法是必须的，因为agent需要知道如何调用动态工具calculate_tip
    否则会报错，提示calculate_tip工具不存在
    """
    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # Handle execution of the dynamic tool
        print("调用wrap_tool_call处理动态工具调用")
        if request.tool_call["name"] == "calculate_tip":
            return handler(request.override(tool=calculate_tip))
        return handler(request)

agent = create_agent(
    model= deep_model,
    tools=[query_weather],
    system_prompt="You are a helpful assistant",
    middleware=[DynamicToolMiddleware()],
)

response = agent.invoke({"messages": [{"role": "user", "content": "计算100元20%税率的小费是多少？"}]})
print(response["messages"][-1].content)
