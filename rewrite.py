from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# template = """Provide a better search query for \
# web search engine to answer the given question, end \
# the queries with ’**’. Question: \
# {x} Answer:"""
template = """你是一个提炼问题的专家，\
你的任务是将用户输入的问题提炼成准确的关键问题。\
如果问题不能提炼，不要对用户的输入进行修改，直接输出用户的原输入，不要输出其他任何文字。\
如果问题能够提炼，输出提炼出问题的内容并保留表示程度的词语，不要输出其他任何文字。\

举例:
用户输入:你能为我详细描述这篇文章提出的模型架构吗?
回答:描述文章提出的模型架构的细节。

用户输入：{x}
回答：
"""


def _parse(text):
    return text.strip('"').strip("**")


def rewrite(query_text):
    rewrite_prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(
        temperature=0,
        model="glm-4",
        openai_api_key="03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    rewriter = rewrite_prompt | llm | StrOutputParser() | _parse

    return rewriter.invoke({"x": query_text})


if __name__ == '__main__':
    distracted_query = "为我描述这篇文章提出的模型架构"
    print(rewrite(distracted_query))
