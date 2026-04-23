from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
import urllib.request
import urllib.error
from langgraph.checkpoint.memory import InMemorySaver
from deepagents import create_deep_agent

from dotenv import load_dotenv
load_dotenv()

@tool
def fetch_text_from_url(url: str) -> str:
    """从指定的url获取内容."""
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("GB2312", errors="replace")
    return text

SYSTEM_PROMPT = """
你是一个专业的文本获取助手，你的任务是根据用户的问题，从指定的url获取内容。
"""

checkpointer = InMemorySaver()

model = init_chat_model(
    model="deepseek:deepseek-chat",
)
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
)

deep_agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[fetch_text_from_url],
)

url = "https://www.zuowen.com/e/20250305/67c82d3cd21ac.shtml"
content = f"""
从url{url}获取作文内容，并总结作文的主要内容
尽量多的回答以下问题
1）完整的作文中有多少行包含子字符串`李老师`（计算行数，而不是行内的出现次数，每行以换行符结束）。
2）完整的作文中包含“李老师”的第一行的行号
3) 一个两句话的中性概要。
在 (1) 和 (2) 上尽力。如果你在任何时候意识到你无法用可用的工具和推理来验证确切答案，不要编造数字：
在该字段中使用 `null` 并在 `how_you_computed_counts` 中说明限制。
如果你遇到任何错误，请报告错误是什么以及错误信息是什么。

"""
response =agent.invoke(
    {"messages": [{"role": "user", "content": content}]}
)

print("agent:", response["messages"][-1].content)
deep_response = deep_agent.invoke({"messages": [{"role": "user", "content": content}]})
print("deep_agent:", deep_response["messages"][-1].content)
