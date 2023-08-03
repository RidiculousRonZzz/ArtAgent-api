import requests
import json

# 定义你要发送的数据
data = {
    "input": "The West Lake",
    "chatbot": [],
    "history": [],
    "userID": 123456
}

try:
    response = requests.post(
        "http://166.111.139.118:22231/gpt4_predict", 
        data=json.dumps(data),  
        headers={'Content-Type':'application/json'}, 
        timeout=900  # 设置超时时间，例如30秒
    )
    response.raise_for_status()  # 如果返回的状态码不是 200，将引发异常
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
else:
    print(response.text)

""" 
时而返回成功，时而不成功

{'chatbot': [['我要画猫', '您可以这样画这幅画：画一只灵动的猫，两只大大的眼睛闪烁着智慧的光芒，身体轻灵，优雅的踩在暗示着深夜的蓝色背景之上。猫咪的身边可以画一些夜间的元素，比如漫天繁星或者明亮的月光，猫咪仿佛在指引着我们走向未知的世界。']],
 'history': [{'role': 'user', 'content': '我要画猫'}, {'role': 'assistant', 'content': '您可以这样画这幅画：画一只灵动的猫，两只大大的眼睛闪烁着智慧的光芒，身体轻灵，优雅的踩在暗示着深夜的蓝色背景之上。猫咪的身边可以画一些夜间的元素，
 比如漫天繁星或者明亮的月光，猫咪仿佛在指引着我们走向未知的世界。'}]}

{'chatbot':[['The West Lake','You could paint this picture like this: The heart of your canvas should be West Lake, shimmering under the golden sun that is just about to set. Inject it with the hues of tranquility - turquoise and sapphire, laced with gold and peach strokes. Capture the reflection of the mist-covered, lush jade mountains in the crystalline water. In the foreground, include a delicate willow tree, its slender branches draping over the lake, creating intricate patterns on the water, and a stone arch bridge echoing tranquility, making a pathway to the traditional pagodas. Fleeting sculls on the sparkling water add a layer of liveness. The painting should whisper a serene song of nature and time, where the present and past merge hauntingly in the tranquil beauty of the West Lake.']],
 'history':[{'role':'user','content':'The West Lake'},
 {'role':'assistant','content':'You could paint this picture like this: The heart of your canvas should be West Lake, shimmering under the golden sun that is just about to set. Inject it with the hues of tranquility - turquoise and sapphire, laced with gold and peach strokes. Capture the reflection of the mist-covered, lush jade mountains in the crystalline water. In the foreground, include a delicate willow tree, its slender branches draping over the lake, creating intricate patterns on the water, and a stone arch bridge echoing tranquility, making a pathway to the traditional pagodas. Fleeting sculls on the sparkling water add a layer of liveness. The painting should whisper a serene song of nature and time, where the present and past merge hauntingly in the tranquil beauty of the West Lake.'}]}
"""