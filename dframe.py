import pandas as pd
from sklearn.cluster import KMeans
import openai
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-4"

# 创建包含文本及其相关属性的DataFrame
data = {
    # 'Screen Content': ['Discuss Van Gogh’s unique style app: ArtChat', 'Experiment with impressionist style app: PaintInspire', 'Analyze the symbolism in War and Peace app: LiteratureAnalyst', '', '', '', 'Photos app: Grand Canyon', '', 'Music app: Bohemian Rhapsody', ''],
    # 'Location': ['', 'Art studio at 123 ArtStreet, Creative City', '', 'Home office at 456 WriteLane, Inspiration Town', 'Living room at 789 RelaxAvenue, Comfort City', '', 'Grand Canyon, Arizona, USA', 'Local gym at 321 FitnessRoad, Health City', '', 'Home at 789 RelaxAvenue, Comfort City'],
    # 'Action': ['discussing', 'painting', 'analyzing', 'writing', 'relaxing', '', 'sightseeing', 'working out', 'listening', 'singing along'],
    # 'User Command': ['Paint a scene in Van Gogh’s style', 'Create an impressionist style landscape', 'Illustrate a symbolic scene from War and Peace', 'Sketch a character for my new book', 'Paint a peaceful living room scene', 'Illustrate a serene night scene', 'Create a landscape painting of the Grand Canyon', 'Draw a dynamic workout scene', 'Paint a scene inspired by Bohemian Rhapsody', 'Sketch a lively sing-along scene'],
    # 'Emotion': ['neutral', 'happiness', 'neutral', 'neutral', 'neutral', 'neutral', 'happiness', 'neutral', 'neutral', 'happiness'],
    'Theme': [
        'A vibrant scene showcasing Van Gogh’s unique style (Work Mode - Art Worker).',
        'A beautiful impressionist style landscape, filled with vibrant colors and brush strokes (Work Mode - Art Worker).',
        'A poignant scene from War and Peace, emphasizing its deep symbolism (Work Mode - Writer).',
        'A detailed sketch of a character, ready to bring life to a new story (Work Mode - Writer).',
        'A peaceful living room scene, encapsulating tranquility and comfort (Life Mode).',
        'A serene night scene, filled with calming hues and soft lights (Life Mode).',
        'A breathtaking landscape painting of the Grand Canyon, capturing its grandeur and natural beauty (Travel Mode).',
        'A dynamic workout scene, showcasing the energy and movement of a gym workout (Motion Mode).',
        'A nostalgic scene inspired by the iconic Bohemian Rhapsody, filled with musical elements and emotions (Music Mode).',
        'A lively sing-along scene, capturing the joy and energy of singing along to a favourite tune (Music Mode).'
    ]
}


df = pd.DataFrame(data)

def create_embedding(query: str):
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return query_embedding_response["data"][0]["embedding"]

# 将每个字段的值串联起来，然后对这个组合的文本进行嵌入
embeddings = [create_embedding(' '.join(row)) for row in df.values]

# 使用PCA将嵌入降至二维
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# 使用KMeans进行聚类
num_clusters = 6
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(embeddings_2d)

# 打印每个文本的所属聚类
df['cluster'] = clustering_model.labels_

# 创建散点图并标出点的序号
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=df['cluster'])
for i in range(len(embeddings_2d)):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], str(i))  # 添加文本标签
plt.show()