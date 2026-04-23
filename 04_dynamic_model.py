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


@wrap_model_call
def dynamic_model_select(request: ModelRequest, handler) -> ModelResponse:
    """根据用户的问题，选择合适的模型."""
    print(request)
    userPrompt = ""
    for message in request.messages:
        if hasattr(message, "content"):
            userPrompt = message.content
            break
    print(f"user prompt: {userPrompt}")
    if "千问" in userPrompt:
        return handler(request.override(model = qianwen_model))
    else:
        return handler(request.override(model = deep_model))


agent = create_agent(
    model=qianwen_model,
    middleware=[dynamic_model_select]
)

response = agent.invoke({"messages": [{"role": "user", "content": "你是deepseek吗？介绍下你自己"}]})
print(response["messages"][-1].content)
