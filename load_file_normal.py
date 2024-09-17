import os
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from zhipuai_embedding import ZhipuAIEmbeddings

ZHIPUAI_API_KEY = "03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg"
os.environ['ZHIPUAI_API_KEY'] = ZHIPUAI_API_KEY


class LoadFileNormal:
    def __init__(self):
        self.flag = 0

    def load_documents(self, data_paths):
        # Initialize PDF loader with specified directory
        document_loader = PyPDFLoader(data_paths)
        # Load PDF documents and return them as a list of Document objects
        return document_loader.load()

    def split_text(self, documents: list[Document], mode="normal"):
        if mode == "normal":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Size of each chunk in characters
                chunk_overlap=200,  # Overlap between consecutive chunks
                length_function=len,  # Function to compute the length of the text
                add_start_index=True,  # Flag to add start index to each chunk
            )
        if mode == "semantic":
            text_splitter = SemanticChunker(
                ZhipuAIEmbeddings(), breakpoint_threshold_type="percentile"
            )

        # Split documents into smaller chunks using text splitter
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        # Print example of page content and metadata for a chunk
        document = chunks[0]
        # print(document.page_content)
        # print(document.metadata)

        return chunks

    def save_to_chroma(self, chunks: list[Document]):
        # Clear out the existing database directory if it exists
        CHROMA_PATH = './chroma'
        if os.path.exists(CHROMA_PATH):
            if self.flag == 0:
                shutil.rmtree(CHROMA_PATH)
            else:
                self.flag = 1
        # 利用huggingface
        # embedding = self.load_embedding()
        embedding = ZhipuAIEmbeddings()
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=CHROMA_PATH,
        )
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
        return db

    def generate_data_store_file(self, file):
        documents = self.load_documents(file)  # Load documents from a source
        print("1")
        chunks = self.split_text(documents, mode="normal")  # Split documents into manageable chunks
        print("2")
        # print(chunks)
        # 111问题
        db = self.save_to_chroma(chunks)  # Save the processed data to a data store
        print("3")
        return db


if __name__ == "__main__":
    CHROMA_PATH = './chroma'
    file_path = './data/BERT.pdf'
    loader = LoadFileNormal()
    vecstore = loader.generate_data_store_file(file_path)
