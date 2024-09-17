import shutil

from unstructured.partition.pdf import partition_pdf
from typing import Any
from pydantic import BaseModel


class load_pdf:
    def __init__(self, filename):
        self.filename = filename
        self.category_counts = {}

    def read_pdf(self):
        # raw_pdf_elements = partition_pdf(
        #     filename=self.filename,
        #     extract_images_in_pdf=False,
        #     # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
        #     # Titles are any sub-section of the document
        #     infer_table_structure=True,
        #     chunking_strategy="by_title",
        #     multipage_sections=False,
        #     max_characters=4000,
        #     new_after_n_chars=3800,
        #     combine_text_under_n_chars=2000,
        #     strategy="hi_res",
        #     use_gpu=True,
        #     include_metadata=True
        # )
        # 每段话都输出分类
        raw_pdf_elements = partition_pdf(
            filename=self.filename,
            strategy="hi_res",
            extract_images_in_pdf=False,
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            use_gpu=True,
            # extract_image_block_types=["Table"],  # optional , "Image"
        )
        return raw_pdf_elements

    def count(self, elements):
        for element in elements:
            category = str(type(element))
            if category in self.category_counts:
                self.category_counts[category] += 1
            else:
                self.category_counts[category] = 1

        # Unique_categories will have unique elements
        unique_categories = set(self.category_counts.keys())
        return unique_categories

    def extract(self, elements):
        class Element(BaseModel):
            type: str
            text: Any

        categorized_elements = []
        for element in elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                categorized_elements.append(Element(type="table", text=str(element)))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                categorized_elements.append(Element(type="text", text=str(element)))

        # Tables
        table_elements = [e for e in categorized_elements if e.type == "table"]
        # print(len(table_elements))

        # Text
        text_elements = [e for e in categorized_elements if e.type == "text"]
        # print(len(text_elements))
        # ---------------------------------------------------------------------------------
        # table_elements = [el for el in elements if el.category == "Table"]
        # text_elements = [el for el in elements if el.category == "CompositeElement"]
        return table_elements, text_elements

    def rerank(self, elements):
        re_element = []
        left = []
        right = []
        page_number = 1
        x1_min = min([el.metadata.coordinates.points[0][0] for el in elements])
        x2_max = max([el.metadata.coordinates.points[1][0] for el in elements])
        mid_line_x_coordinate = (x2_max + x1_min) / 2
        for element in elements:
            if element.metadata.page_number == page_number:
                if element.metadata.coordinates.points[0][0] < mid_line_x_coordinate:
                    left.append(element)
                else:
                    right.append(element)
            else:
                left.sort(key=lambda z: z.metadata.coordinates.points[0][1])
                right.sort(key=lambda z: z.metadata.coordinates.points[0][1])
                page_number = page_number + 1
                re_element = re_element + left
                re_element = re_element + right
                left = []
                right = []
                if element.metadata.coordinates.points[0][0] < mid_line_x_coordinate:
                    left.append(element)
                else:
                    right.append(element)
        return re_element

    def process_pdf(self):
        elements = self.read_pdf()
        re_elements = self.rerank(elements)
        # 清洗数据，只要其中的Title和NarrativeText
        wash_elements = []
        for element in re_elements:
            # print(type(element))
            # IPython.embed()
            if 'unstructured.documents.elements.Title' in str(type(element)):
                wash_elements.append(element)
            elif 'unstructured.documents.elements.NarrativeText' in str(type(element)):
                if len(element.text) > 10:
                    wash_elements.append(element)

        # 初始化变量
        page_number = 1
        paragraph_number = 1
        list_content = []

        i = 0
        while i < len(wash_elements):
            element = wash_elements[i]

            # 根据元素类型构建 page_content
            if 'unstructured.documents.elements.NarrativeText' in str(type(element)):
                if element.metadata.page_number == page_number:
                    page_content = f"Page is {element.metadata.page_number},paragraph is {paragraph_number},content is {element.text}"
                    paragraph_number += 1
                else:
                    page_number = element.metadata.page_number
                    paragraph_number = 1
                    page_content = f"Page is {page_number},paragraph is {paragraph_number},content is {element.text}"
                    paragraph_number += 1

                # 检查下一个元素是否满足条件
                if i + 1 < len(wash_elements):
                    next_element = wash_elements[i + 1]
                    if next_element.text[0] >= 'a' and next_element.text[0] <= 'z':
                        page_content += next_element.text
                        i += 1  # 手动跳过下一个元素

            else:
                page_content = f"{element.text}"

            # print(page_content)
            # 手动增加 i
            i += 1

            list_content.append(page_content + '\n')
        return list_content

    def store(self):
        import os
        from zhipuai_embedding import ZhipuAIEmbeddings
        from langchain_chroma import Chroma
        chunks = self.process_pdf()  # Load documents from a source
        print("1")
        # print(chunks)
        # 111问题
        CHROMA_PATH = './chroma'
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        # 利用huggingface
        # embedding = self.load_embedding()
        embedding = ZhipuAIEmbeddings()
        db = Chroma.from_texts(
            chunks,
            embedding=embedding,
            persist_directory=CHROMA_PATH,
        )
        print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    import os
    ZHIPUAI_API_KEY = "03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg"
    os.environ['ZHIPUAI_API_KEY'] = ZHIPUAI_API_KEY
    file_path = "./data/BERT.pdf"
    load_document = load_pdf(file_path)
    x = load_document.process_pdf()
    load_document.store()
    from ConversationRetrieval import Conversation
    chat = Conversation()
    query = str(input("shuru"))
    chat.query_rag_langchain_history_new(query)
"""
    elements = load_document.read_pdf()
    print(len(elements))
    re_elements = load_document.rerank(elements)
    # 清洗数据，只要其中的Title和NarrativeText
    wash_elements = []
    for element in re_elements:
        # print(type(element))
        if 'unstructured.documents.elements.Title' in str(type(element)):
            wash_elements.append(element)
        elif 'unstructured.documents.elements.NarrativeText' in str(type(element)):
            wash_elements.append(element)
    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # set_len = set()
    # for element in elements:
    #     # print(type(element))
    #     if 'unstructured.documents.elements.Title' in str(type(element)):
    #         length = abs(element.metadata.coordinates.points[0][1]-element.metadata.coordinates.points[2][1])
    #         set_len.add(length)
    # print(set_len)
    for element in wash_elements:
        print(type(element), element.text)
"""
