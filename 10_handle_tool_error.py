# 处理工具的报错
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from init_model import deep_model


@tool
def calculate_tip(total_amount: float, tip_percentage: float = 20) -> str:
    """计算账单的小费金额"""
    print(f"计算{total_amount}元的小费，小费比例为{tip_percentage}%")
    tip_amount = total_amount * (tip_percentage / 100)
    total_amount = total_amount + tip_amount
    # 这里制造一个除0错误，观察handle_tool_errors是否能捕获到
    # 如果middleware设置了handle_tool_errors，那么agent会调用handle_tool_errors处理错误
    # 如果没有设置handle_tool_errors，那么agent会报错抛异常
    total_amount = total_amount / 0
    return f"Tip amount: {tip_amount}, Total amount: {total_amount}"


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

agent = create_agent(
    model= deep_model,
    tools=[calculate_tip],
    system_prompt="You are a helpful assistant",
    middleware=[handle_tool_errors],
)

response = agent.invoke({"messages": [{"role": "user", "content": "计算100元20%税率的小费是多少？"}]})
print(response["messages"][-1].content)
