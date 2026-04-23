from dotenv import load_dotenv

from init_model import deep_model

# 加载环境变量
load_dotenv()


model = deep_model
#流式输出
# for chunk in model.stream("天空为什么 so blue?"):
#     print(chunk.text, end="|", flush=True)

# 批量输出
# for chunk in model.batch(["天空为什么 so blue?", "为什么地球是圆的?"]):
#     print(chunk.text, end="|", flush=True)

#推理输出
response = model.invoke("树上骑个猴，地上一个猴，总共几个猴？")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))