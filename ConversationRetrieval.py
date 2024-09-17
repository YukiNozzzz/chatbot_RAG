import os
import fitz
import gradio as gr
from PIL import Image
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from zhipuai_embedding import ZhipuAIEmbeddings
from rewrite import rewrite
ZHIPUAI_API_KEY = "03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg"
os.environ['ZHIPUAI_API_KEY'] = ZHIPUAI_API_KEY

COHERE_API_KEY = 'zcWoQyAya82ynN1A9kLZntKiez7YogggCnpjdcH7'
os.environ['COHERE_API_KEY'] = COHERE_API_KEY


class Conversation:
    def __init__(self):
        self.store = {}
        self.page = 0

    def add_text(self, history, text):
        # 将用户的询问加到历史消息中去
        if not text:
            raise gr.Error('Enter text')
        history.append((text, ''))
        return history

    def render_file(self, file):
        # 加载pdf文档后，显示特定的页数。
        doc = fitz.open(file.name)
        page = doc[self.page]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    # work!!!
    def query_rag_langchain_history(self, query_texts):
        # YOU MUST - Use same embedding function as before
        print("11111111111111")
        # 利用huggingface的来加载embedding
        # embedding_function = self.load_embedding()
        # 利用zhipuai
        query_texts = rewrite(query_texts)
        print(query_texts)
        CHROMA_PATH = './chroma'
        embedding_function = ZhipuAIEmbeddings()
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        retriever = vectorstore.as_retriever()  #search_kwargs={"k": 10}

        # cohere_reranker = CohereRerank(
        #     model="rerank-english-v3.0",
        #     top_n=3,
        #     cohere_api_key=COHERE_API_KEY)
        # retriever = ContextualCompressionRetriever(
        #     base_compressor=cohere_reranker,
        #     base_retriever=retriever,
        # )

        llm = ChatOpenAI(
            temperature=0,
            model="glm-4",
            openai_api_key="03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
        )
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Your task is to use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\
        Questions are about the content of paper and recognize the title or auther of the paper. \
        The title and author of paper always written at the beginning of the paper. \
    
        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",

        )

        result = conversational_rag_chain.invoke(input={"input": query_texts},
                                                 config={"configurable": {"session_id": "abc123"}})
        print(result)

        return result

    def query_rag_langchain_history_new(self, query_texts):
        print("222222222")
        CHROMA_PATH = './chroma'
        embedding_function = ZhipuAIEmbeddings()
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        retriever = vectorstore.as_retriever()  #
        template = """
                You are an assistant tasked with summarizing tables and text.
                Provide a concise summary of the following content, which includes page and paragraph information.
                Content:{context}
                Question:{input}
                """
        llm = ChatOpenAI(
            temperature=0,
            model="glm-4",
            openai_api_key="03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
        )
        prompt = ChatPromptTemplate.from_template(template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({"input": query_texts})
        print(response["answer"])

        return result

    def query(self, query_text, context):
        print("222222222")
        query_text = rewrite(query_text)
        print(query_text)
        template = """
        你需要回答一系列关于文章结构以及文章内容的问题，文本和句子的索引均从1开始。\
        文本的每段间由"\n"隔开，每句话由"."结束。\
        基于以下文本内容给出精确的回答。
        文本内容：{context}
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        llm = ChatOpenAI(
            temperature=0,
            model="glm-4",
            openai_api_key="03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg",
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
        )
        qa_chain = prompt | llm
        chain = RunnableWithMessageHistory(
            qa_chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        result = chain.invoke(input={"question": query_text, "context": context},
                              config={"configurable": {"session_id": "abc123"}})

        return result


if __name__ == "__main__":
    chatbot = Conversation()
    while True:
        query_texts = str(input("请输入："))
        result = chatbot.query_rag_langchain_history(query_texts)
        print(result["answer"])
