from ConversationRetrieval import Conversation
from intent_recognition import Intent_cls
from load_file_pdf import load_pdf

chatbot = Conversation()


class RAG_pipline:
    def __init__(self):
        self.intro_doc = None
        self.ab_doc = None
        self.con_doc = None

    def query_rag(self, query_text, history):
        global result
        res = Intent_cls(query_text)
        res = res.content
        if res == "文章内容" or res == "闲聊":
            result = chatbot.query_rag_langchain_history(query_text)
            result = result["answer"]
        elif res == "文章结构":
            if "introduction" in query_text.lower():
                result = chatbot.query(query_text, self.intro_doc)
                result = result.content
            elif "abstract" in query_text.lower():
                result = chatbot.query(query_text, self.ab_doc)
                result = result.content
            elif "conclusion" in query_text.lower():
                result = chatbot.query(query_text, self.con_doc)
                result = result.content
        for char in result:
            history[-1][-1] += char
        return history, " "

    def store_doc(self, file_path):
        print(11)
        loadpdf = load_pdf(file_path)
        elements = loadpdf.read_pdf()
        re_elements = loadpdf.rerank(elements)
        # 清洗数据，只要其中的Title和NarrativeText
        wash_elements = []
        for element in re_elements:
            # print(type(element))
            if 'unstructured.documents.elements.Title' in str(type(element)):
                wash_elements.append(element)
            elif 'unstructured.documents.elements.NarrativeText' in str(type(element)):
                wash_elements.append(element)
        intro_doc = ""
        ab_doc = ""
        con_doc = ""
        ab_flag = 0
        intro_flag = 0
        con_flag = 0
        for element in wash_elements:
            if 'unstructured.documents.elements.Title' in str(type(element)) and "introduction" in element.text.lower():
                intro_flag = 1
                intro_doc += element.text + "\n"
            if 'unstructured.documents.elements.Title' in str(type(element)) and "relatedwork" in element.text.replace(
                    " ", "").lower():
                intro_flag = 0
            if intro_flag == 1 and len(element.text) > 30:
                intro_doc += element.text + "\\n"
            if 'unstructured.documents.elements.Title' in str(type(element)) and "abstract" in element.text.lower():
                ab_flag = 1
                ab_doc += element.text + "\n"
            if 'unstructured.documents.elements.Title' in str(type(element)) and "introduction" in element.text.lower():
                ab_flag = 0
            if ab_flag == 1 and len(element.text) > 30:
                ab_doc += element.text + "\\n"
            if 'unstructured.documents.elements.Title' in str(type(element)) and "conclusion" in element.text.lower():
                con_flag = 1
                con_doc += element.text + "\n"
            if 'unstructured.documents.elements.Title' in str(type(element)) and "references" in element.text.lower():
                con_flag = 0
            if con_flag == 1 and len(element.text) > 30:
                con_doc += element.text + "\\n"
        self.intro_doc = intro_doc
        self.ab_doc = ab_doc
        self.con_doc = con_doc
        print("store done!!!")
        return intro_doc, ab_doc, con_doc


if __name__ == '__main__':
    file_path = './data/BERT.pdf'
    rag_pipline = RAG_pipline()
    intro_doc, ab_doc, con_doc = rag_pipline.store_doc(file_path)
    print(intro_doc)
    print(ab_doc)
    print(con_doc)
