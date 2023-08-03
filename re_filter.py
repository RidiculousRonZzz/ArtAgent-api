import re

def filter_context(text, vector):  # 对空格不敏感，但一定要用英文的逗号
    sections = ["Location", "Phone-Content", "Facial Expression", "Weather", "Music", "User command"]
    text_parts = re.split("(Location:|Phone-Content:|Facial Expression:|Weather:|Music:|User command:)", text)
    
    new_text_parts = []
    for i in range(1, len(text_parts), 2):
        section = text_parts[i][:-1]
        content = text_parts[i+1].split(',')[0] if i+1 < len(text_parts) else text_parts[i+1]
        if (section != "User command" and vector[sections.index(section)] == 1 and content != "[]") or section == "User command":
            new_text_parts.append(section + ':' + content)
    
    return ','.join(new_text_parts)


text1 = "Location:[北京市颐和园], Phone-Content:[], Facial Expression:[],Weather:[晴],Music:[],User command:[根据我的地点画幅油画]"
vector1 = [1,0,1,1,1]
print(filter_context(text1, vector1))  # "Location:[北京市颐和园],Weather:[晴],User command:[根据我的地点画幅油画]"

text2 = "Location:[北京市798艺术区],Phone-Content:[],Facial Expression:[],Weather:[小雨],Music:[],User command:[画一幅抽象派画作表示反战情绪]"
vector2 = [0,0,1,1,1]
print(filter_context(text2, vector2))  # "Weather:[小雨],User command:[画一幅抽象派画作表示反战情绪]"