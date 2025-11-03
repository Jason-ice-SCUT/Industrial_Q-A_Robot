import gradio as gr
import os_mirror.os_mirror
import RAG.query

title = "Industrial Q&A Robot"
description = "This is a chatbot that can answer questions based on the provided specific industrial context."

QA_ROBOT = gr.Interface(
    fn = RAG.query.QA_Generate, 
    inputs = "text", 
    outputs = "text",
    title = title, 
    description = description,
    examples = [
        ["电动平衡车的安全要求是什么？"],
        ["工业机器人在制造业中的应用有哪些？"],
        ["如何提高生产线的自动化水平？"]]
)

# Launch the interface
if __name__ == "__main__":
    QA_ROBOT.launch()

