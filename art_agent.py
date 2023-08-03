import gradio as gr
from utils import *

greetings = [("Hello!", "Hello! I am ArtAgent ChatBot, an AI assistant that engages with you on art concepts, generates, and modifies images.\n\n I utilize GPT4, VisualGLM-6B, and Stable Diffusion models.\n\n If you wish to generate creative images based on art discussions, please click the 'Generate Creative Image' button. If you aim to meticulously modify an image, please click the 'Please Edit It!' button. If you are not satisfied, feel free to regenerate. If you wish to start a new creative theme, please click the 'Begin a New Topic' button.")]

gr.Chatbot.postprocess = postprocess

with gr.Blocks(title="ArtAgent ChatBot") as demo:
    gr.HTML("""<h1 align="center"> 🎊 ArtAgent  ChatBot 🎊 </h1>""")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(greetings).style(height=600)
            with gr.Box():
                with gr.Row():
                    with gr.Column(scale=2):
                        user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=3).style(container=False)
                    with gr.Column(scale=1, min_width=100):
                        submitBtn = gr.Button("Chat with ArtAgent 🚀",)
                        emptyBtn = gr.Button("Begin a New Topic",)
            with gr.Row():
                sd_width = gr.Slider(512, 1024, value=768, step=32, label="width", interactive=True)
                sd_height = gr.Slider(512, 1024, value=768, step=32, label="height", interactive=True)
        with gr.Column(scale=3):
            with gr.Group():
                with gr.Tab("Gallery"):
                    result_gallery = gr.Gallery(label='Output', show_label=False).style(preview=True)
                with gr.Tab("Upload Image"):
                    upload_image = gr.Image(label='Upload', brush_radius=30, show_label=True, interactive=True, type="pil", tool='color-sketch')
            with gr.Row():
                drawBtn = gr.Button("Generate Creative Image 🎨", variant="primary")
                editBtn = gr.Button("Please Edit It!", variant="primary")
            with gr.Tab("Sketchpad"):
                sketchpad = gr.Sketchpad(shape=(1000, 1000), brush_radius=5, type="pil", tool="color-sketch")

                # pil 调粗细，color-sketch 是画板颜色选取
                # brush_radius 是笔触粗细，gr.Sketchpad().style(height=280, width=280) 没用，只是扩大画板外框框的大小

    history = gr.State([])
    result_list = gr.State([])
    userID = gr.State(random.randint(0, 9999999))  # 用户在未刷新情况下随机给到一个id
    cnt = gr.State(0)

    def click_count():
        cnt = 0
        yield cnt

    submitBtn.click(gpt4_predict, [user_input, chatbot, history, userID], [chatbot, history], show_progress=True)  # 艺术讨论
    submitBtn.click(reset_user_input, [], [user_input])  # 发送完信息就清空。一次点击触发两个函数
    submitBtn.click(click_count, [], [cnt])  # 一次生图不满意，继续点击按钮，中间没说话：向他道歉

    editBtn.click(gpt4_sd_edit, [chatbot, history, result_list, userID, cnt, sd_width, sd_height], [chatbot, history, result_list, result_gallery, cnt, result_gallery], show_progress=True)

    drawBtn.click(gpt4_sd_draw, [chatbot, history, result_list, userID, cnt, sd_width, sd_height], [chatbot, history, result_list, result_gallery, cnt, result_gallery], show_progress=True)

    upload_image.change(read_image, [upload_image, chatbot, history, userID], [chatbot, history], show_progress=True)

    emptyBtn.click(reset_state, [chatbot, userID], [chatbot, history, cnt], show_progress=True)


os.makedirs('output', exist_ok=True)
demo.queue().launch(share=True, inbrowser=True, server_name='127.0.0.1', server_port=6006, favicon_path="./favicon.ico")