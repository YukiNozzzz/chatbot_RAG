import os
from interface import create_demo
from ConversationRetrieval import Conversation
from load_file_normal import LoadFileNormal
from RAG_pipline import RAG_pipline


chatbot = Conversation()
loader = LoadFileNormal()
rag_pipline = RAG_pipline()
ZHIPUAI_API_KEY = "03075038f931c2ebddbce60dbf318e09.JoPtPiCtZYG3P6kg"
os.environ['ZHIPUAI_API_KEY'] = ZHIPUAI_API_KEY
demo, chat_history, show_img, txt, submit_button, uploaded_pdf = create_demo()
# 目前无后处理，直接切分的效果较好
with demo:
    # 传入文件后，将加载的文件可视化，输出图片显示
    uploaded_pdf.upload(chatbot.render_file, inputs=[uploaded_pdf], outputs=[show_img]). \
        success(rag_pipline.store_doc, inputs=[uploaded_pdf])
    uploaded_pdf.upload(loader.generate_data_store_file, inputs=[uploaded_pdf])

    # 前一个事件成功运行结束后，再执行success的
    # .click第一个参数是fn，传入一个函数，当触发成功，调用这个函数
    # input是我们刚才输入的txt,接着append到我们的chat_history上,输出就是append后的chat_history

    submit_button.click(chatbot.add_text, inputs=[chat_history, txt], outputs=[chat_history], queue=False). \
        success(rag_pipline.query_rag, inputs=[txt, chat_history], outputs=[chat_history, txt])
    # submit_button.click(query_rag, inputs=[txt], outputs=[chat_history, txt])

if __name__ == "__main__":
    demo.launch()
