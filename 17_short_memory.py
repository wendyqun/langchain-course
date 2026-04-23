from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from init_model import deep_model

model = deep_model

memory = InMemorySaver()

agent = create_agent(
    model=model,
    checkpointer=memory
)

_config= {"configurable": {"thread_id": "1"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "你好, 我是大尾巴狼"}]},
    config=_config
)

response = agent.invoke(
    {
        "messages": [{"role": "user", "content": "我是谁"}],
    }
    , config=_config
)
print(response["messages"][-1].content)
