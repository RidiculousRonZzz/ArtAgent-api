import requests
import json
from utils import ImageRequest, HistoryItem

history = [
    {'role': 'user', 'content': 'The West Lake'}, 
    {'role': 'assistant', 'content': 'You could paint this picture like this: The heart of your canvas should be West Lake, shimmering under the golden sun that is just about to set. Inject it with the hues of tranquility - turquoise and sapphire, laced with gold and peach strokes. Capture the reflection of the mist-covered, lush jade mountains in the crystalline water. In the foreground, include a delicate willow tree, its slender branches draping over the lake, creating intricate patterns on the water, and a stone arch bridge echoing tranquility, making a pathway to the traditional pagodas. Fleeting sculls on the sparkling water add a layer of liveness. The painting should whisper a serene song of nature and time, where the present and past merge hauntingly in the tranquil beauty of the West Lake.'}
]
history_data = [HistoryItem.parse_obj(item).dict() for item in history]

# url = 'http://127.0.0.1:8000/gpt4_sd_draw'
url = 'http://166.111.139.116:22231/gpt4_sd_draw'
data = ImageRequest(
    chatbot = [['The West Lake', 'You could paint this picture like this: The heart of your canvas should be West Lake, shimmering under the golden sun that is just about to set. Inject it with the hues of tranquility - turquoise and sapphire, laced with gold and peach strokes. Capture the reflection of the mist-covered, lush jade mountains in the crystalline water. In the foreground, include a delicate willow tree, its slender branches draping over the lake, creating intricate patterns on the water, and a stone arch bridge echoing tranquility, making a pathway to the traditional pagodas. Fleeting sculls on the sparkling water add a layer of liveness. The painting should whisper a serene song of nature and time, where the present and past merge hauntingly in the tranquil beauty of the West Lake.'],],
    history = history_data,
    userID=12345,
    cnt=0,
    width=768,
    height=768
)
try:
    response = requests.post(url, data=json.dumps(data.dict()), headers={'Content-Type':'application/json'}, timeout=100000)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
else:
    print(response.text)

# http://localhost:8000/static/images/3eabaa98-0e94-43a8-88a3-000371d9e5fb.png
# http://166.111.139.116:22231/static/images/144fbd72-9e24-4a92-908e-250eab2a1ce2.png