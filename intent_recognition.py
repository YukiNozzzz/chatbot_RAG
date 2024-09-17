from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def Intent_cls(query_text):
    template = """
    你是一位聊天机器人，专注于识别用户的意图，并根据意图执行对应操作。用户的意图可以是以下几种之一：闲聊、文章结构、文章内容。\
    一个用户的输入只对应一个意图。

    **意图类型：**
    1. 闲聊：用户在表达问候、礼貌或开启对话时或问与文章内容无关的问题时选择该意图。
    2. 文章结构：用户要求总结或概述某个特定章节时选择该意图。
    3. 文章内容：用户询问或讨论某个特定的文章内容时选择该意图。

    **示例对话：**
    用户输入：“你好，今天过得怎么样？”
    意图：闲聊

    用户输入：“他刚才第二句话说了什么？”
    意图：闲聊

    用户输入：“M1系列可以跑iPhone和iPad游戏吗？”
    意图：闲聊

    用户输入：“这篇文章的实验结果怎么样?”
    意图：文章内容
    
    用户输入：“这篇文章讲了什么？”
    意图：文章内容
    
    用户输入：“这篇文章的题目是什么？”
    意图：文章内容
    
    用户输入：“这篇文章提出的算法是什么？”
    意图：文章内容
    
    用户输入：“帮我总结下这篇文章”
    意图：文章内容
    
    用户输入：“为我介绍一下模型的架构”
    意图：文章内容
    
    用户输入：“这篇文章的Abstract讲了什么？”
    意图：文章结构

    用户输入：“这篇文章Introduction第一段的第一句话是什么？”
    意图：文章结构

    用户输入：“概括下Introduction的第一段讲了什么？”
    意图：文章结构

    用户输入：“related work的第二段讲了什么？”
    意图：文章结构

    ---
    用户输入:{input}
    请根据用户的输入，只输出识别到意图的名字。

    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(
        temperature=0,
        model="glm-4",
        openai_api_key="03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )
    chain = {"input": RunnablePassthrough()} | prompt | model
    result = chain.invoke(query_text)
    return result


if __name__ == '__main__':
    while True:
        query_text = str(input("请输入:"))
        res = Intent_cls(query_text)
        print(res)
