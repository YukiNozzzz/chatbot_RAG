import os
import shutil
from langchain_chroma import Chroma
from zhipuai_embedding import ZhipuAIEmbeddings
from load_file_md import DocumentManager
from langchain.schema import Document

ZHIPUAI_API_KEY = "03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg"
os.environ['ZHIPUAI_API_KEY'] = ZHIPUAI_API_KEY


class ChromaManager:
    def __init__(self, all_splits, all_metadata, persist_directory='./chroma'):
        self.all_splits = all_splits
        self.all_metadata = all_metadata
        self.persist_directory = persist_directory
        self.vectordb = None

    # Method to create and persist embeddings
    def create_chroma(self):
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        embedding = ZhipuAIEmbeddings()
        # 将刚才读取后切分得到的splits和metadata存入数据库中
        # self.vectordb = Chroma.from_texts(texts=self.all_splits, metadatas=self.all_metadata, embedding=embedding,
        #                                   persist_directory=self.persist_directory)

        doc = []
        for split, metadata in zip(self.all_splits, self.all_metadata):
            doc.append(Document(page_content=split, metadata=metadata))
        self.vectordb = Chroma.from_documents(documents=doc, persist_directory="./chroma", embedding=embedding)

        print(f"Successfully saved {len(self.all_splits)} chunks into db!!!")
        return self.vectordb


if __name__ == '__main__':
    # file_paths = "./md/BERT.md"
    # load_documents = DocumentManager(file_paths)
    # load_documents.load_documents()
    # all_splits, all_metadata = load_documents.split_documents()
    #
    # vectordb = ChromaManager(all_splits, all_metadata)
    # vectorstore = vectordb.create_chroma()
    vectorstore = Chroma(persist_directory="./chroma", embedding_function=ZhipuAIEmbeddings())
    from langchain_openai import ChatOpenAI

