# 使用pydantic定义输出 Schema

from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from init_model import deep_model


# 1. 定义输出 Schema
class ProductReview(BaseModel):
    """商品评论分析结果"""
    rating: int | None = Field(description="评分，1-5 分", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="情感倾向")
    key_points: list[str] = Field(description="关键要点，每条 1-3 个词")

# 2. 创建 Agent
agent = create_agent(
    model=deep_model,
    response_format=ProductReview
)

# 3. 调用
response = agent.invoke({
    "messages": [{"role": "user", "content": "分析这条评论：'超棒的产品，五星好评！'"}]
})
print(response)
print(response["messages"][-1].content)
print(response["structured_response"])
