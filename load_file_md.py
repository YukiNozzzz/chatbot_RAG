from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.documents = None
        self.all_sections = []

    def load_documents(self):
        loader = TextLoader(self.file_path, encoding='utf8')
        markdown_document = loader.load()
        self.documents = markdown_document[0].page_content

    # 先用md切分器切分，再用迭代切分器切分
    def split_documents(self):
        headers_to_split_on = [("##", "Section"), ("###", "Subsection")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(self.documents)

        chunk_size = 250
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_splits = []
        all_metadata = []
        for header_group in md_header_splits:
            # print(1)
            _splits = text_splitter.split_text(header_group.page_content)
            _metadata = [header_group.metadata for _ in _splits]
            all_splits += _splits
            all_metadata += _metadata
        return all_splits, all_metadata


if __name__ == '__main__':
    file_paths = "./md/Trans.md"
    load_documents = DocumentManager(file_paths)
    load_documents.load_documents()
    all_splits, all_metadata = load_documents.split_documents()
    for i, s in zip(all_splits, all_metadata):
        print(i)
        print(s)
