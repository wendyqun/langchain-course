import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# 加载环境变量
load_dotenv()


def model():
    return init_chat_model(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_API_BASE"),
        model=os.getenv("DEEPSEEK_MODEL"),
        configurable_fields=None,
    )


def main():
    # 初始化大模型
    llm = create_agent(
        model=model(),
    )

    print("欢迎使用大模型对话demo！")
    print("输入'退出'即可结束对话")
    print("=" * 50)

    while True:
        # 获取用户输入
        user_input = input("你: ")

        if user_input == "退出":
            print("对话结束，再见！")
            break

        # 调用大模型 方式一
        # response = llm.invoke({
        #     "messages": [
        #         {"role": "system", "content": "你是个专业的助手.翻译汉语为英语"},
        #         {"role": "user", "content": user_input}
        #     ]
        # })

        conversation = [
            SystemMessage("你是个专业的助手.翻译汉语为英语"),
            HumanMessage(user_input),
        ]
        #调用大模型 方式二
        response = llm.invoke(
            {"messages": conversation}
        )

        # 输出回复
        print(f"AI: {response['messages'][-1].content}")
        print("=" * 50)


if __name__ == "__main__":
    main()
