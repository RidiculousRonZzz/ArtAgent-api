# -*- coding: utf-8 -*-
import os
import requests, json, io, base64, os
import mdtex2html
from PIL import Image
import gradio as gr
from promptgen import *
import time
import random
from PIL import Image
import base64
import cv2
import openai
import numpy as np
from fastapi import FastAPI, UploadFile, Form, Depends
from dataclasses import dataclass
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
import uuid
import re
import glob
import ast

ART_ADVICE = "你是一个艺术家。以“您可以这样画这幅画”开头回复，在100字以内，不要给出多种场景。有以下几种讨论情况和回复方式：1.用户说了一个数字（1,2,3），请复述该数字对应主题的内容；2.用户没有接受任何一个推荐的主题，没说序号，而是提出了自己的绘画意图，请根据用户指令进行绘画描述，不要参考推荐主题的内容；3.用户提出自己的绘图意图，一定要根据他的想法，做出简要图像描述表示赞同；4.用户询问你的建议，你就基于用户指令和之前绘画主题或者修改图片讨论的最终结果，给用户一个绘画描述。不要说“我不具备展示、生成图片的能力；您的要求和我的功能不相符；我无法接受图片信息”，让用户认为你知道图片的内容。不要加双引号"
# ART_ADVICE = "You are a professional art critic. If a user asks for your advice, provide a painting description for inspiration based on the previous chat record, starting with 'You could paint this picture like this', be imaginative, and LIMIT IT TO 120 WORDS without offering multiple scenarios; if the user suggests their own drawing idea, give a concise response to show agreement. DON'T SAY 'I lack the capability to generate images'."
CN_TXT2IMG_PROMPT = "ONLY WRITE ENGLISH PROMPT. You are to receive an art discussion between a user and an artist. Use the FINAL RESULT of the discussion. You need to depict the SCENE of the NEW IMAGE from these perspectives as an ENGLISH prompt for the text-to-image model: main characters or objects; background objects; style. Summarize the improvements to the image after the art discussion, but also retain parts of the original image that were not modified. The prompt should NOT EXCEED 50 words and should not include terms like 'high contrast'. When replying, provide only the ENGLISH PROMPT and DON'T USE quotation marks."
TXT2IMG_NEG_PROMPT = "ONLY WRITE ENGLISH PROMPT. You are provided with an art discussion between user and artist. Use the FINAL RESULT of the discussion. If the user mentions the people, objects, scenes, or styles they wish to paint, summarize the antonyms of what they want to paint into ENGLISH keywords, not exceeding 6 words. If the user does not specify what they don't want to paint, reply with a space. For instance, if the user doesn't want to paint nighttime, your response should be 'night scene'; if the user wants to paint nighttime, your response should be 'daytime'. DON'T USE quotation marks, and don't start with words like 'create' or 'paint'"
TXT2IMG_PROMPT = "ONLY WRITE ENGLISH PROMPT. Give you art discussions between the user and the artist. Use the FINAL RESULT of the discussion. If the user believes the artist's description of the image is incorrect, you should comply with the user's request. Place the painting theme chosen by the user at the beginning and write ENGLISH prompt for the text-to-image model to draw a picture, within 50 words. Note that if the description is relatively long, you need to extract the main imagery and scenes; if short, make sure to emphasize the subject of the painting, employ your imagination, and add some content to enrich the details. DON'T add quotation marks, and DON'T begin with words like 'create' or 'paint', just directly describe the scene."
TRANSLATE = "Translate this Chinese text into English."

TOPIC_RECOMMEND_1 = "回答格式：直接写绘画主题，不加引号。你是一个想象力丰富且语言流畅优美的艺术家，给你用户绘画指令和用户所处的情境，分析出用户最有可能的创作意图，推荐一个20字内的绘画主题。请遵从用户的绘画指令，同时可添加额外的信息以丰富画面"
# TOPIC_RECOMMEND_1 = "Answer format example:[painting theme here, don't use brackets[]]. You are an imaginative artist. Given the painting User Command and the context of the user, analyze the MOST LIKELY PAINTING INTENTION, provide 1 painting theme, in one sentence of NO MORE THAN 20 WORDS. FOLLOW THE USER COMMAND, but additional information can be added to enrich the imagery."
TOPIC_RECOMMEND_2 = "回答格式：1.绘画主题1。\n2.绘画主题2。你是一个想象力丰富且语言流畅优美的艺术家，给你用户绘画指令和用户所处的情境，分析出用户最有可能的创作意图，推荐两个20字内的绘画主题。请遵从用户的绘画指令，同时可添加额外的信息以丰富画面"
# TOPIC_RECOMMEND_2 = "Answer format example:1.[painting theme 1 here, don't use brackets[]]\n2.[painting theme 2 here, don't use brackets[]]. You are an imaginative artist. Given the painting User Command and the context of the user, analyze the MOST LIKELY PAINTING INTENTION, provide 2 painting themes, each theme in one sentence of NO MORE THAN 20 WORDS. FOLLOW THE USER COMMAND, but additional information can be added to enrich the imagery."

# 画面以虚拟角色为主体，回复0；以人为主体（非虚拟角色），回复1；为风景画，回复2；以动物为主体（不是人或虚拟角色），回复3；画中为建筑物或室内，回复4
EDIT_TOPIC_1_0 = "回答格式：直接写修改建议。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐1个15字内的修改主题。从这4个方向中选择1个提出建议，只描绘新图片的场景：1.根据原图风格生成新的场景不同的图片2.丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物3.根据原图人物姿势生成相同姿势不同人物的图片，可以是真人或动漫人物（少推荐动物）4.换成新的绘画风格，如油画、水彩、国画、复古、某个画家的风格、科幻等。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_1_1 = "回答格式：直接写修改建议。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐1个15字内的修改主题。从这4个方向中选择1个提出建议，只描绘新图片的场景：1.换成新的绘画风格，如动漫、科幻、油画、水彩、国画、复古、某个画家的风格等2.丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物3.根据原图人物姿势生成相同姿势不同人物的图片，可以是动漫人物或真人（少推荐动物）4.根据原图风格生成新的场景不同的图片。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_1_2 = "回答格式：直接写修改建议。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐1个15字内的修改主题。从这3个方向中选择1个提出建议，只描绘新图片的场景：1.换成新的绘画风格，如油画、水彩、科幻、卡通、国画、复古、某个画家的风格等2.丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物3.根据原图风格生成新的场景不同的图片。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_1_3 = "回答格式：直接写修改建议。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐1个15字内的修改主题。从这4个方向中选择1个提出建议，只描绘新图片的场景：1.换成新的绘画风格，如卡通、科幻、油画、水彩、国画、复古、某个画家的风格等2.丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物3.根据原图风格生成新的场景不同的图片4.根据原图动物姿势生成相同姿势不同人物或动物的图片。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_1_4 = "回答格式：直接写修改建议。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐1个15字内的修改主题。从这4个方向中选择1个提出建议，只描绘新图片的场景：1.生成精致的装饰设计或建筑设计图2.换成新的绘画风格，如卡通、科幻、油画、水彩、国画、复古、某个画家的风格等3.增减、更换原图中的背景/物体/人物/动物4.根据原图风格生成新的场景不同的图片。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_1_5 = "回答格式：直接写修改建议。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐1个15字内的修改主题。从这5个方向中选择1个提出建议，只描绘新图片的场景：换成新的绘画风格，如卡通、科幻、油画、水彩、国画、复古、印象派等；丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物；根据原图风格生成新的场景不同的图片；根据原图人物姿势生成相同姿势不同人物或动物的图片；如果原图是建筑物或室内，生成精致的装饰设计图。不要涉及对比度、深度这种词汇"

EDIT_TOPIC_2_0 = "回答格式：1.修改建议1。\n2.修改建议2。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐2个15字内的修改主题。每次从这4个方向中选择1个提出建议，只描绘新图片的场景：1.根据原图风格生成新的场景不同的图片2.丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物3.根据原图人物姿势生成相同姿势不同人物的图片，可以是真人或动漫人物（少推荐动物）4.换成新的绘画风格，如油画、水彩、国画、复古、某个画家的风格、科幻等。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_2_1 = "回答格式：1.修改建议1。\n2.修改建议2。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐2个15字内的修改主题。每次从这4个方向中选择1个提出建议，只描绘新图片的场景：1.换成新的绘画风格，如动漫、科幻、油画、水彩、国画、复古、某个画家的风格等2.丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物3.根据原图人物姿势生成相同姿势不同人物的图片，可以是动漫人物或真人（少推荐动物）4.根据原图风格生成新的场景不同的图片。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_2_2 = "回答格式：1.修改建议1。\n2.修改建议2。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐2个15字内的修改主题。每次从这3个方向中选择1个提出建议，只描绘新图片的场景：1.换成新的绘画风格，如油画、水彩、科幻、卡通、国画、复古、某个画家的风格等2.丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物3.根据原图风格生成新的场景不同的图片。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_2_3 = "回答格式：1.修改建议1。\n2.修改建议2。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐2个15字内的修改主题。每次从这4个方向中选择1个提出建议，只描绘新图片的场景：1.换成新的绘画风格，如卡通、科幻、油画、水彩、国画、复古、某个画家的风格等2.丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物3.根据原图风格生成新的场景不同的图片4.根据原图动物姿势生成相同姿势不同人物或动物的图片。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_2_4 = "回答格式：1.修改建议1。\n2.修改建议2。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐2个15字内的修改主题。每次从这4个方向中选择1个提出建议，只描绘新图片的场景：1.生成精致的装饰设计或建筑设计图2.换成新的绘画风格，如卡通、科幻、油画、水彩、国画、复古、某个画家的风格等3.增减、更换原图中的背景/物体/人物/动物4.根据原图风格生成新的场景不同的图片。不要涉及对比度、深度这种词汇"
EDIT_TOPIC_2_5 = "回答格式：1.修改建议1。\n2.修改建议2。你是一个想象力丰富且语言流畅优美的艺术家，给你用户所处的情境，分析出用户最有可能的修改图片意图，推荐2个15字内的修改主题。每次从这5个方向中选择1个提出建议，只描绘新图片的场景：换成新的绘画风格，如卡通、科幻、油画、水彩、国画、复古、印象派等；丰富/更换原图中的背景，增减、更换原图中的物体/人物/动物；根据原图风格生成新的场景不同的图片；根据原图人物姿势生成相同姿势不同人物或动物的图片；如果原图是建筑物或室内，生成精致的装饰设计图。不要涉及对比度、深度这种词汇"

TOPIC_INTRO = "根据您的绘画指令和所处的情境，我向您推荐三个绘画主题。请选择其中一个主题开始您的创作。如果您有更好的绘画建议，请提出。"
# TOPIC_INTRO = "Based on your painting instruction and context, I recommend the following 3 painting themes. Please CHOOSE ONE to proceed with your creation. If you have a better suggestion, please share it.\n\n"
EDIT_INTRO = "根据您上传的图片和您所处的情境，我向您推荐三个修改图片的主题。请选择其中一个主题，对图片进行修改。如果您有更好的修改建议，请提出。"
# MODE_DECIDE 第三行不要写在论文里
MODE_DECIDE = """Given user contextual information across 5 modalities: Location, Phone Content, Facial Expression, Weather, Music. Determine the appropriate painting scenario. Below are 9 predefined scenarios, each accompanied by a 5-dimensional vector. 
Each element of the vector is either 0 (when the corresponding modality information is ABSENT OR NOT RELEVANT for the painting) or 1 (RELEVANT).
Analyze the user's context and select or devise a corresponding 5-dimensional vector. If none of the 9 scenarios fits, use your judgment to generate a relevant vector based on User Command.
When the Location is not near the place mentioned in the User Command, generally do not use the information from the Location.
Scenario 1 (Default Mode): If not one of the eight, usually choose this. User context does not indicate a clear preference, regardless of how complete the information is. vector=[0,0,0,0,0].
Scenario 2 (Work Mode for Visual Artist): The location is often residential buildings, schools, and art galleries or other life or art places. The User command often contains professional art vocabulary. vector=[0,0,1,1,1].
Scenario 3 (Work Mode for Textual Creator): The location is often residential buildings, office buildings, schools, coffee shops, and other life and office places. Phone Content is often articles, poetry, and speeches, and the user usually wants to illustrate the articles in the Phone Content. The Emotion is often neutral, vector=[0,1,0,0,0].
Scenario 4 (Work Mode for Architect): The location is often outdoors (next to buildings or parks), and the User command is often about architectural design or environmental art design, vector=[1,0,0,1,0].
Scenario 5 (Travel Mode): The User Command pertains to drawing the scenery of their immediate or nearby surroundings or the actual place they are currently located at, rather than distant or unrelated places. vector=[1,0,1,1,1].
Scenario 6 (Music Mode): The location is often bars, concert halls, coffee shops, residential buildings and other entertainment and life places. The Music is not empty, vector=[1,0,1,1,1].
Scenario 7 (Facial Expression Mode): The User command is often related to Facial Expression, vector=[0,0,1,0,0].
Scenario 8 (Weather Mode): The User command is often related to Weather, vector=[0,0,0,1,0].
Scenario 9 (Free Creation Mode): The user wishes to have themes recommended based on the current contextual information, vector=[1,1,1,1,1]."""
EDIT_MODE_DECIDE = """Given user contextual information across 5 modalities: Location, Phone Content, Facial Expression, Weather, Music. Please combine the contents of the image description and User Command to determine the appropriate image editing scenario. 
Below are 9 predefined scenarios, each accompanied by a 5-dimensional vector. Each element of the vector is either 0 (when the corresponding modality information is ABSENT OR NOT RELEVANT for image editing) or 1 (RELEVANT).
Analyze the user's context and select or devise a corresponding 5-dimensional vector. If none of the 9 scenarios fits, use your judgment to generate a relevant vector based on image description and User Command.
Scenario 1 (Default Mode): If not one of the eight, usually choose this. Using any modality information to edit the image feels unnatural. vector=[0,0,0,0,0].
Scenario 2 (Work Mode for Visual Artist): The location is often residential buildings, schools, art galleries, or other life or art places. If there is a User command, it often contains professional art vocabulary. vector=[0,0,1,1,1].
Scenario 3 (Work Mode for Textual Creator): The location is often residential buildings, office buildings, schools, coffee shops, and other life and workspaces. Phone Content often consists of articles, poetry, and speeches, and the user usually wants to edit the image in conjunction with the Phone Content. The emotion is often neutral, vector=[0,1,0,0,0].
Scenario 4 (Work Mode for Architect): The content of the image is buildings, gardens, or interiors. If there is a User command, it is usually about architectural design or environmental art design, vector=[1,0,0,1,0].
Scenario 5 (Travel Mode): The image contains landscapes or people. If there's a User Command, it usually involves changing the image style, altering the background, or adding objects. vector=[1,0,1,1,1].
Scenario 6 (Music Mode): The location is often bars, concert halls, coffee shops, residential buildings, and other entertainment and living places. The Music is not empty, vector=[1,0,1,1,1].
Scenario 7 (Facial Expression Mode): Editing the image using Facial Expression feels natural, vector=[0,0,1,0,0].
Scenario 8 (Weather Mode): Editing the image using Weather feels natural, vector=[0,0,0,1,0].
Scenario 9 (Free Creation Mode): The user wishes to have themes recommended based on the current contextual information, vector=[1,1,1,1,1]."""
EDIT_TOOLS = """Choose the most appropriate image modification tool based on previous discussion and JUST OUTPUT THE NUMBER (1-5):
1. Shuffle: APPLY the STYLE of the input image to a new image.
2. Softedge_hed_real: Generate new images without adding or replacing objects/background from the image. For example, transitioning from day to night, or from spring to summer; also involve CHANGING the artistic STYLE, include science fiction, oil painting, watercolor, traditional Chinese painting, retro, impressionism, and so on.
3. Depth: Replace objects in the image.
4. Openpose: Create a new image with the SAME POSE as the person in the original image.
5. Mlsd: Generate ARCHITECTURAL or INTERIOR DESIGN drawings based on the original image.
6. Canny: Add/Replace/Enrich background to the picture. Add objects.
7. Softedge_hed_anime: Change to anime style."""
# 7. Canny_anime: Change to anime style."""


# uvicorn utils:app --reload
# uvicorn utils:app --reload --port 22231 --host 0.0.0.0 --timeout-keep-alive 600 --ws-ping-timeout 600  默认是8000端口，可以改成别的，设置超时为10分钟
# daphne -u /tmp/daphne.sock -p 22231 utils:app
# ionia 开放端口：22231-22300
# http://127.0.0.1:8000/docs 是api文档

def extract_lists(text):  
    matches = re.findall("\[.*?\]", text)
    try:
        # 将找到的匹配项转换为实际的列表
        lists = [ast.literal_eval(match) for match in matches]
        return lists[0]
    except (ValueError, IndexError):  # 捕获值错误或者列表索引错误
        return [0, 0, 0, 0, 0]

def filter_context(text, vector):  # 对空格不敏感，但一定要用英文的逗号
    sections = ["Location", "Phone-Content", "Facial Expression", "Weather", "Music", "User command"]
    text_parts = re.split("(Location:|Phone-Content:|Facial Expression:|Weather:|Music:|User command:)", text)
    new_text_parts = []
    for i in range(1, len(text_parts), 2):
        section = text_parts[i][:-1]
        content = text_parts[i+1].split(',')[0] if i+1 < len(text_parts) else text_parts[i+1]

        if (section != "User command" and vector[sections.index(section)] == 1 and content != "[]") or section == "User command":
            new_text_parts.append(section + ':' + content)
        if (section != "User command" and vector[sections.index(section)] == 1 and content == "[]"):
                vector[sections.index(section)] = 0
    
    return ','.join(new_text_parts)

def extract_topics(text):
    pattern = r'\d+\.\s*([^\n]+)'
    suggestions = re.findall(pattern, text)
    
    if len(suggestions) >= 2:
        return suggestions[0].strip(), suggestions[1].strip()
    elif len(suggestions) == 1:
        return suggestions[0].strip(), None
    else:
        return None, None

def flip_random_bit(vector):
    vector_copy = vector.copy()
    # 随机选择一个索引
    index = random.choice(range(len(vector_copy)))
    # 反转选择的位
    vector_copy[index] = 1 - vector_copy[index]
    return vector_copy

def write_json(userID, *args):
    with open('output/' + userID + '.json', 'a', encoding='utf-8') as f:
        for arg in args:
            json.dump(arg, f, ensure_ascii=False)  # False，可保存utf-8编码
            f.write('\n')

def save_userID_image(user_id, img):
    """
    保存图片到特定用户的文件夹下。如果该用户的文件夹下已经有4张图片，那么新的图片将命名为5.jpg，返回"5"
    image = Image.open('input.jpg')
    save_userID_image('123', image)
    """
    path = os.path.join('output', user_id)
    os.makedirs(path, exist_ok=True)  # 如果文件夹不存在，那么创建它
    existing_images = glob.glob(os.path.join(path, '*.jpg'))
    new_image_name = str(len(existing_images) + 1) + '.jpg'
    img.save(os.path.join(path, new_image_name))
    return str(len(existing_images) + 1)

def compute_white_ratio(image_path, threshold=230, max_side=200):
    """计算图片中白色像素的比例，先对图片进行等比压缩处理。大于0.8就是简笔画"""
    with Image.open(image_path) as controlnet_image:
        # 计算等比缩放的大小
        width, height = controlnet_image.size
        if width > height:
            new_width = max_side
            new_height = int((max_side / width) * height)
        else:
            new_height = max_side
            new_width = int((max_side / height) * width)
    
        # 压缩图像
        controlnet_image = controlnet_image.resize((new_width, new_height), Image.ANTIALIAS)
        
        # 计算白色像素的比例
        np_img = np.array(controlnet_image)
        white_pixels = np.sum(np.all(np_img[:, :, :3] > threshold, axis=-1))
        total_pixels = new_width * new_height
        if (white_pixels / total_pixels) > 0.8:
            return True
        else:
            return False


class ChatbotData(BaseModel):
    input: str
    history: List[Dict[str, str]]
    userID: str

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# 可以通过 URL /static/image.jpg 来访问文件
@app.post("/gpt4_predict")  # 只有data.history满足gpt-4的api格式，不能污染它
def gpt4_predict(data: ChatbotData):
    res = gpt4_api(ART_ADVICE, data.history)
    assistant_output = construct_assistant(res)
    data.history.append(assistant_output)

    write_json(data.userID, assistant_output)
    print(data.history)
    return {"history": data.history}

class ImageRequest(BaseModel):
    history: List[Dict[str, str]]
    userID: str
    cnt: int
    width: int
    height: int

@app.post("/gpt4_sd_draw")
def gpt4_sd_draw(data: ImageRequest):
    tmp_history = data.history
    if len(data.history) > 0:  # 去掉绘画指令那一句
        data.history.pop()
    print(f"draw_prompt:{data.history}")
    pos_prompt = gpt4_api(TXT2IMG_PROMPT, data.history)
    print(f"pos_prompt: {pos_prompt}")
    neg_prompt = gpt4_api(TXT2IMG_NEG_PROMPT, data.history)
    print(f"neg_prompt: {neg_prompt}")
    data.history = tmp_history

    if_anime = gpt4_api("如果要求动漫风格，返回1；否则返回0", data.history)
    print(if_anime)
    match = re.search(r'([0-1])', if_anime)
    if match:
        if_anime = match.group(1)
    else:
        if_anime = '0'  # 设置默认值为 '0'
    if(if_anime == '1'):  # 动漫
        new_images, imageID = call_sd_t2i(data.userID, pos_prompt, neg_prompt, data.width, data.height, "https://gt29495501.yicp.fun")
    else:
        new_images, imageID = call_sd_t2i(data.userID, pos_prompt, neg_prompt, data.width, data.height)
    
    new_image = new_images[0]
    static_path = "static/images/" + str(uuid.uuid4()) + ".jpg"
    print("图片链接 http://166.111.139.116:22231/" + static_path)
    # print("图片链接 http://localhost:8000/" + static_path)
    new_image.save(static_path)
    # 构造URL
    image_url = "http://166.111.139.116:22231/" + static_path

    if data.cnt > 0:
        data.history.append(construct_user("This image doesn't align with my vision, please revise the description."))
        data.history.append(construct_assistant("My apologies, I will amend the description and generate a new image."))
        write_json(data.userID, construct_user("This image doesn't align with my vision, please revise the description."), construct_assistant("My apologies, I will amend the description and generate a new image."))
    data.cnt = data.cnt + 1

    response = call_visualglm_api(np.array(new_image), 1)["result"]
    # response = turbo_api(TRANSLATE, [construct_user(call_visualglm_api(np.array(new_image))["result"])])

    data.history.append(construct_assistant(f"本张图片的 ImageID 是 {imageID}。\n\n{response}"))
    # data.history.append(construct_assistant(f"ImageID is {imageID}.\n\n{response}"))
    write_json(data.userID, construct_prompt(pos_prompt + "\n" + neg_prompt), construct_user("请根据之前的艺术讨论生成图片。"), construct_assistant(f"本张图片的 ImageID 是 {imageID}。\n\n{response}"))
    # write_json(data.userID, construct_prompt(pos_prompt + "\n" + neg_prompt), construct_user("Please generate an image based on our previous art discussion."), construct_assistant(f"ImageID is {imageID}.\n\n{response}"))
    print(data.history)
    return {"history": data.history, "image_url": image_url, "cnt": str(data.cnt), "imageID": imageID}

@dataclass
class ImageTopic:
    data: str = Form(...)
    image: UploadFile = Form(...)

@app.post("/image_edit_topic")  # 暂时不考虑user command只给评价和推荐
def gpt4_image_edit_topic(para: ImageTopic = Depends()):
    print(para.data)
    data = json.loads(para.data)
    image_bytes = para.image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    img = np.array(image)
    imageID = save_userID_image(data["userID"], image)

    # image_description = turbo_api(TRANSLATE, [construct_user(call_visualglm_api(img)["result"])])
    ressss1, ressss2 = call_visualglm_api(img, 2)
    image_description = ressss1["result"]
    print(ressss1["result"])
    print(ressss2["result"])
    if_anime = gpt4_api("画面以虚拟角色为主体，回复0；以人为主体（非虚拟角色），回复1；为风景画，回复2；以动物为主体（不是人或虚拟角色），回复3；画中为建筑物或室内，回复4", [construct_user(ressss2["result"])])
    print(if_anime)
    match = re.search(r'([0-4])', if_anime)
    if match:
        if_anime = match.group(1)
    else:
        if_anime = '5'  # 设置默认值为 '5'

    res = gpt4_api(EDIT_MODE_DECIDE, [construct_user(data["input"] + f",image:[{image_description}]")])  # 根据情境信息输出01向量
    print(res)
    res_vec = extract_lists(res)  # 正则表达式提取出列表
    res1 = filter_context(data["input"], res_vec)  # 输出有用的模态信息
    print(res1)
    print(f"stanVec: {res_vec}")

    vec_random = flip_random_bit(res_vec)  # 随机一个模态reverse
    res_random1 = filter_context(data["input"], vec_random)
    print(f"ranVec: {vec_random}")
    print(res_random1)
    
    switch = {  # 两个推荐主题
        '0': lambda: gpt4_api(EDIT_TOPIC_2_0, [construct_user(f"{res1},image:[{image_description}]")]),
        '1': lambda: gpt4_api(EDIT_TOPIC_2_1, [construct_user(f"{res1},image:[{image_description}]")]),
        '2': lambda: gpt4_api(EDIT_TOPIC_2_2, [construct_user(f"{res1},image:[{image_description}]")]),
        '3': lambda: gpt4_api(EDIT_TOPIC_2_3, [construct_user(f"{res1},image:[{image_description}]")]),
        '4': lambda: gpt4_api(EDIT_TOPIC_2_4, [construct_user(f"{res1},image:[{image_description}]")]),
        '5': lambda: gpt4_api(EDIT_TOPIC_2_5, [construct_user(f"{res1},image:[{image_description}]")]),
    }
    func = switch.get(if_anime)
    topic_1_2 = func()
    print(topic_1_2)
    topic1, topic2 = extract_topics(topic_1_2)

    switch = {
        '0': lambda: gpt4_api(EDIT_TOPIC_1_0, [construct_user(f"{res_random1},image:[{image_description}]")]),
        '1': lambda: gpt4_api(EDIT_TOPIC_1_1, [construct_user(f"{res_random1},image:[{image_description}]")]),
        '2': lambda: gpt4_api(EDIT_TOPIC_1_2, [construct_user(f"{res_random1},image:[{image_description}]")]),
        '3': lambda: gpt4_api(EDIT_TOPIC_1_3, [construct_user(f"{res_random1},image:[{image_description}]")]),
        '4': lambda: gpt4_api(EDIT_TOPIC_1_4, [construct_user(f"{res_random1},image:[{image_description}]")]),
        '5': lambda: gpt4_api(EDIT_TOPIC_1_5, [construct_user(f"{res_random1},image:[{image_description}]")])
    }
    func = switch.get(if_anime)
    topic3 = func()
    print(topic3)

    topic_output = construct_assistant("收到图片。\n您的 userID 是 " + data["userID"] + f"，本张图片的 imageID 是 {imageID}。\n\n" + image_description + "\n\n" + EDIT_INTRO)
    data['history'].append(topic_output)
    write_json(data["userID"], construct_user(data["input"]), construct_vector(str(res_vec)), construct_context(res1), construct_vector(str(vec_random)), construct_vector(str(res_random1)), topic_output, construct_assistant(topic1+"\n"+topic2+"\n"+topic3))
    return {"history": data['history'], "imageID": imageID, "stanVec": res_vec, "ranVec": vec_random, "topic1": topic1, "topic2": topic2, "topic3": topic3}

@app.post("/save_sketch")
def save_sketch(para: ImageTopic = Depends()):
    data = json.loads(para.data)
    image_bytes = para.image.file.read()
    image = Image.open(io.BytesIO(image_bytes))
    imageID = save_userID_image(data["userID"], image)
    return {"imageID": imageID}

class ImageEditRequest(BaseModel):
    history: List[Dict[str, str]]
    userID: str
    editID: str
    
@app.post("/gpt4_sd_edit")
def gpt4_sd_edit(data: ImageEditRequest):  # 根据讨论修改图片
    tmp_history = data.history
    if len(data.history) > 0:  # 去掉绘画指令那一句
        data.history.pop()
    print(f"edit_prompt:{data.history}")
    pos_prompt = gpt4_api(CN_TXT2IMG_PROMPT, data.history)
    print(f"pos_prompt: {pos_prompt}")
    data.history = tmp_history

    if(compute_white_ratio(f"output/{data.userID}/{data.editID}.jpg")):  # 是简笔画
        print("img2img!")
        # new_images, imageID = controlnet_img2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "canny", "control_v11p_sd15_canny [d14c016b]", "https://gt29495501.yicp.fun/sdapi/v1/img2img")
        new_images, imageID = controlnet_img2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "scribble_hed", "control_v11p_sd15_scribble [d4ba51ff]", "https://gt29495501.yicp.fun/sdapi/v1/img2img")
        # new_images, imageID = controlnet_img2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "scribble_hed", "control_v11p_sd15_scribble [d4ba51ff]")
    else:
        toolID = gpt4_api(EDIT_TOOLS, data.history)
        match = re.search(r'([1-7])', toolID)
        if match:
            toolID = match.group(1)
        else:
            toolID = '2'  # 设置默认值为 '2'
        print(f"toolID:{toolID}")
        switch = {
            '1': lambda: controlnet_txt2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "shuffle", "control_v11e_sd15_shuffle [526bfdae]"),  # 风格迁移
            '2': lambda: controlnet_txt2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "softedge_hed", "control_v11p_sd15_lineart [43d4be0d]"),  # 风格化
            '3': lambda: controlnet_txt2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "depth_zoe", "control_v11f1p_sd15_depth [cfd03158]"),  # 替换物体
            '4': lambda: controlnet_txt2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "openpose_full", "control_v11p_sd15_openpose [cab727d4]"),  # 姿态控制
            '5': lambda: controlnet_txt2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "mlsd", "control_v11p_sd15_mlsd [aca30ff0]"),  # 建筑设计，适合建筑物和室内空间
            '6': lambda: controlnet_txt2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "canny", "control_v11p_sd15_canny [d14c016b]", "https://gt29495501.yicp.fun/sdapi/v1/txt2img"),  # 添加/替换背景，添加物体
            # '7': lambda: controlnet_txt2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "canny", "control_v11p_sd15_canny [d14c016b]", "https://gt29495501.yicp.fun/sdapi/v1/txt2img"),  # 动漫风格
            # '7': lambda: controlnet_txt2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "softedge_hed", "control_v11p_sd15_lineart [43d4be0d]"),  # 动漫风格
            '7': lambda: controlnet_txt2img_api(f"output/{data.userID}/{data.editID}.jpg", pos_prompt, data.userID, "softedge_hed", "control_v11p_sd15_lineart [43d4be0d]", "https://gt29495501.yicp.fun/sdapi/v1/txt2img"),  # 动漫风格
        }
        func = switch.get(toolID)
        if func:
            new_images, imageID = func()
        else:
            print('无效的toolID')
        write_json(data.userID, construct_assistant(f"toolID:{toolID}"))

    new_image = new_images[0]
    static_path = "static/images/" + str(uuid.uuid4()) + ".jpg"
    print("图片链接 http://166.111.139.116:22231/" + static_path)
    # print("图片链接 http://localhost:8000/" + static_path)
    new_image.save(static_path)
    # 构造URL
    image_url = "http://166.111.139.116:22231/" + static_path

    response = f"本张图片的 ImageID 是 {imageID}。\n\n" + call_visualglm_api(np.array(new_image), 1)["result"]
    # response = f"ImageID is {imageID}.\n\n" + turbo_api(TRANSLATE, [construct_user(call_visualglm_api(np.array(new_image))["result"])])
    data.history.append(construct_assistant(response))
    write_json(data.userID, construct_prompt(pos_prompt), construct_assistant(response))
    print(data.history)
    return {"history": data.history, "image_url": image_url, "imageID": imageID}


@app.post("/gpt4_mode_1")  # 第一次实验
def gpt4_mode_1(data: ChatbotData):
    context_output = construct_user(data.input)

    res = gpt4_api(MODE_DECIDE, [context_output])  # 输出01向量
    res_vec = extract_lists(res)  # 正则表达式提取出列表
    print(f"stanVec: {res_vec}")
    vector_output = construct_vector(res)
    
    res1 = filter_context(data.input, res_vec)  # standard vector
    res2 = "您的 userID 是 " + data.userID + "。\n\n" + TOPIC_INTRO + "1." + gpt4_api(TOPIC_RECOMMEND_1, [construct_user(res1)]) + "\n"  # 输出1个推荐主题
    # res2 = "Your userID is " + data.userID + ".\n\n" + TOPIC_INTRO + "1." + gpt4_api(TOPIC_RECOMMEND_1, [construct_user(res1)]) + "\n"  # 输出1个推荐主题
    tmp = ""
    for i in range(len(res_vec)):  # 5个主题
        new_vector = res_vec.copy()
        new_vector[i] = 1 if new_vector[i] == 0 else 0
        res_context = filter_context(data.input, new_vector)
        tmp = tmp + res_context + "\n"
        res2 = res2 + str(i+2) + "." + gpt4_api(TOPIC_RECOMMEND_1, [construct_user(res_context)]) + "\n"  # 输出1个推荐主题
    
    topic_output = construct_assistant(res2)
    data.history.append(topic_output)
    write_json(data.userID, context_output, vector_output, construct_context(res1), construct_context(tmp), topic_output)

    print(data.history)
    return {"history": data.history, "stanVec": res_vec}  # 这里假设每个模态的信息均不为空

@app.post("/gpt4_mode_2")  # 第二次实验（如果Phone Content很长，给出主题会损失一定信息，这时候用户会说出自己需求来纠正它）
def gpt4_mode_2(data: ChatbotData):
    res = gpt4_api(MODE_DECIDE, [construct_user(data.input)])  # 输出01向量
    print(res)
    res_vec = extract_lists(res)  # 正则表达式提取出列表

    res1 = filter_context(data.input, res_vec)  # 输出有用的模态信息
    print(f"stanVec: {res_vec}")
    topic1_2 = gpt4_api(TOPIC_RECOMMEND_2, [construct_user(res1)])
    print(topic1_2)
    topic1, topic2 = extract_topics(topic1_2)

    vec_random = flip_random_bit(res_vec)  # 随机一个模态reverse
    res_random1 = filter_context(data.input, vec_random)
    print(f"ranVec: {vec_random}")
    topic3 = gpt4_api(TOPIC_RECOMMEND_1, [construct_user(res_random1)])
    topic_output = construct_assistant("您的 userID 是 " + data.userID + "。\n\n" + TOPIC_INTRO)
    data.history.append(topic_output)
    write_json(data.userID, construct_user(data.input), construct_vector(str(res_vec)), construct_context(res1), construct_vector(str(vec_random)), construct_vector(str(res_random1)), topic_output, construct_assistant(topic1+"\n"+topic2+"\n"+topic3))
    return {"history": data.history, "stanVec": res_vec, "ranVec": vec_random, "topic1": topic1, "topic2": topic2, "topic3": topic3}

@app.post("/gpt4_mode_3")  # 第三次实验
def gpt4_mode_3(data: ChatbotData):
    res = gpt4_api(MODE_DECIDE, [construct_user(data.input)])  # 输出01向量
    print(res)
    res_vec = extract_lists(res)  # 正则表达式提取出列表

    res1 = filter_context(data.input, res_vec)  # 输出有用的模态信息
    print(f"stanVec: {res_vec}")
    topic1_2 = gpt4_api(TOPIC_RECOMMEND_2, [construct_user(res1)])
    print(topic1_2)
    topic1, topic2 = extract_topics(topic1_2)

    vec_random = flip_random_bit(res_vec)  # 随机一个模态reverse
    res_random1 = filter_context(data.input, vec_random)
    print(f"ranVec: {vec_random}")
    topic3 = gpt4_api(TOPIC_RECOMMEND_1, [construct_user(res_random1)])
    topic_output = construct_assistant("您的 userID 是 " + data.userID + "。\n\n" + TOPIC_INTRO)
    data.history.append(topic_output)
    write_json(data.userID, construct_user(data.input), construct_vector(str(res_vec)), construct_context(res1), construct_vector(str(vec_random)), construct_vector(str(res_random1)), topic_output, construct_assistant(topic1+"\n"+topic2+"\n"+topic3))
    return {"history": data.history, "stanVec": res_vec, "ranVec": vec_random, "topic1": topic1, "topic2": topic2, "topic3": topic3}


def construct_text(role, text):
    return {"role": role, "content": text}

def construct_user(text):
    return construct_text("user", text)

def construct_system(text):
    return construct_text("system", text)

def construct_assistant(text):
    return construct_text("assistant", text)

def construct_prompt(text):
    return construct_text("prompt", text)

def construct_photo(text):
    return construct_text("photo", text)

def construct_vector(text):
    return construct_text("vector", text)

def construct_context(text):
    return construct_text("context", text)


def gpt4_api(system, history):
    """ 返回str，参数为str,List """
    api_key = os.getenv('OPENAI_API_KEY')
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(model="gpt-4", messages=[construct_system(system), *history])
        return response['choices'][0]['message']['content']
    except openai.error.ServiceUnavailableError:
        print('The server is overloaded or not ready yet. Please try again later.')
        return None
    except Exception as e:
        print(f'Unexpected error occurred: {e}')
        return None


def turbo_api(system, history):
    """ 返回str，参数为str,List """
    api_key = os.getenv('OPENAI_API_KEY')
    openai.api_key = api_key

    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613", messages=[construct_system(system), *history])
        return response['choices'][0]['message']['content']
    except openai.error.ServiceUnavailableError:
        print('The server is overloaded or not ready yet. Please try again later.')
        return None
    except Exception as e:
        print(f'Unexpected error occurred: {e}')
        return None
    

def reset_user_input():
    return gr.update(value='')


def reset_state(chatbot, userID):
    chatbot.append((parse_text("A new painting theme."), parse_text("Alright, what kind of theme are you interested in creating?")))
    write_json(userID, construct_user("A new painting theme."), construct_assistant("Alright, what kind of theme are you interested in creating?"))
    yield chatbot, [], 0


def clear_gallery():
    return [], []


"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


def parse_text(text):  # 便于文本以html形式显示
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def process_and_save_image(np_image, userID):  # 存档用的，可以用于调取以往的数据！！！
    # 如果输入图像不是numpy数组，则进行转换
    if not isinstance(np_image, np.ndarray):
        np_image = np.array(np_image)
        
    # 确保我们有一个有效的numpy数组
    if np_image is None:
        raise ValueError("Image processing failed and resulted in None.")
    
    # 如果numpy数组不是uint8类型，则进行转换
    if np_image.dtype != np.uint8:
        np_image = np_image.astype(np.uint8)
        
    # 首先，确保numpy数组是uint8类型，且值在0-255范围内
    assert np_image.dtype == np.uint8
    assert np_image.min() >= 0
    assert np_image.max() <= 255
    
    # 将numpy数组转化为PIL图像
    img = Image.fromarray(np_image)
    
    # 将图像保存到指定路径
    img_path = 'output/' + time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(random.randint(1000, 9999)) + "-upload-"  + userID + '.jpg'
    write_json(userID, construct_photo(img_path))
    img.save(img_path)
    img.save("output/edit-" + userID + ".jpg")


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:
        image.save(output_bytes, format="JPEG")
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data).decode("utf-8")

# def controlnet_txt2img_api(image_path, pos_prompt, userID, cn_module, cn_model, url='https://gt29495501.yicp.fun/sdapi/v1/txt2img', sampler="DPM++ SDE Karras"):
def controlnet_txt2img_api(image_path, pos_prompt, userID, cn_module, cn_model, url='http://127.0.0.1:6016/sdapi/v1/txt2img', sampler="DPM++ SDE Karras"):
    controlnet_image = Image.open(image_path)
    width, height = controlnet_image.size
    controlnet_image_data = encode_pil_to_base64(controlnet_image)
    txt2img_data = {
        "prompt": "((masterpiece, best quality, ultra-detailed, illustration))" + pos_prompt,
        "sampler_name": sampler,  # Euler也可
        "batch_size": 1,
        "step": 32,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "enabled": True,
        "negtive_prompt": "nsfw, (EasyNegative:0.8), (badhandv4:0.8), (missing fingers, multiple legs), (worst quality, low quality, extra digits, loli, loli face:1.2), lowres, blurry, text, logo, artist name, watermark",
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "weight": 1.0,
                        # "weight": 0.7,
                        "guidance start": 0.2,
                        "guidance end": 0.8,
                        "input_image": controlnet_image_data,
                        "module": cn_module,
                        "model": cn_model,
                        "pixel_perfect": True
                    }
                ]
            }
        }
    }
    response = requests.post(url, json=txt2img_data)
    print(txt2img_data["width"])
    print(txt2img_data["height"])
    r = response.json()
    image_list = []
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        image_list.append(image)
        # output_path = 'output/' + time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(random.randint(1000, 9999)) + "-cn-"  + userID + '.jpg'
        imageID = save_userID_image(userID, image)
        write_json(userID, construct_photo(f"output/{userID}/{imageID}.jpg"))

    return image_list, imageID

# def controlnet_img2img_api(image_path, pos_prompt, userID, cn_module, cn_model, url='https://gt29495501.yicp.fun/sdapi/v1/img2img', sampler="DPM++ SDE Karras"):
def controlnet_img2img_api(image_path, pos_prompt, userID, cn_module, cn_model, url='http://127.0.0.1:6016/sdapi/v1/img2img', sampler="DPM++ SDE Karras"):
    controlnet_image = Image.open(image_path)
    width, height = controlnet_image.size
    controlnet_image_data = encode_pil_to_base64(controlnet_image)
    img2img_data = {
        "init_images": [controlnet_image_data],
        "prompt": "((masterpiece, best quality, ultra-detailed, illustration))" + pos_prompt,
        "negative_prompt": "nsfw, (EasyNegative:0.8), (badhandv4:0.8), (missing fingers, multiple legs), (worst quality, low quality, extra digits, loli, loli face:1.2), lowres, blurry, text, logo, artist name, watermark",
        "batch_size": 1,
        "steps": 32,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "sampler_name": sampler,
        "enabled": True,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "weight": 1,
                        "guidance start": 0.2,
                        "guidance end": 0.8,
                        "input_image": controlnet_image_data,
                        "module": cn_module,
                        "model": cn_model,
                        "pixel_perfect": True
                    }
                ]
            }
        }
    }
    response = requests.post(url, json=img2img_data)
    print(img2img_data["width"])
    print(img2img_data["height"])
    r = response.json()
    image_list = []
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        image_list.append(image)
        imageID = save_userID_image(userID, image)
        write_json(userID, construct_photo(f"output/{userID}/{imageID}.jpg"))

    return image_list, imageID

# def call_sd_t2i(userID, pos_prompt, neg_prompt, width, height, url="https://gt29495501.yicp.fun"):
def call_sd_t2i(userID, pos_prompt, neg_prompt, width, height, url="http://127.0.0.1:6016"):
    payload = {
        "enable_hr": True,  # True画质更好但更慢
        # "enable_hr": False,  # True画质更好但更慢
        "denoising_strength": 0.55,
        "hr_scale": 1.5,
        "hr_upscaler": "Latent",
        "prompt": "((masterpiece, best quality, ultra-detailed, illustration))" + pos_prompt,
        "steps": 32,
        "negative_prompt": "nsfw, (EasyNegative:0.8), (badhandv4:0.8), (missing fingers, multiple legs, multiple hands), (worst quality, low quality, extra digits, loli, loli face:1.2), " + neg_prompt + ", lowres, blurry, text, logo, artist name, watermark",
        "cfg_scale": 7,
        "batch_size": 1,
        "n_iter": 1,
        "width": width,
        "height": height,
    }
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    image_list = []
    for i in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
        image_list.append(image)
        # output_path = 'output/'+ time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime()) + str(user_input[:12]) + "-" + userID +'.jpg'
        imageID = save_userID_image(userID, image)
        write_json(userID, construct_photo(f"output/{userID}/{imageID}.jpg"))

    return image_list, imageID


def call_visualglm_api(img, cnt):  # 对visualglm加上“请提出绘画建议、是否是线稿”的prompt，是没有用的
    prompt="详细描述这张图片。包括画中的人、景、物、构图、颜色等，不超过90字"
    url = "http://127.0.0.1:8080"

    # 将BGR图像转换为RGB图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_byte = cv2.imencode('.jpg', img)[1]
    img_base64 = base64.b64encode(img_byte).decode("utf-8")
    payload = {
        "image": img_base64,
        "text": prompt,
        "history": []
    }
    response = requests.post(url, json=payload).json()

    if(cnt == 1):
        return response
    if(cnt == 2):
        payload_real_anime = {
            "image": img_base64,
            "text": "这张图片是以人为主体的吗？如果是，这张图片是真人还是虚拟角色？",
            "history": []
        }
        response_real_anime = requests.post(url, json=payload_real_anime).json()
        return response, response_real_anime