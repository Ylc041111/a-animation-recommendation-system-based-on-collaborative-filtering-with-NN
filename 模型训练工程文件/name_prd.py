import csv
import json
import os
import random
import shutil
import tempfile
from nltk.corpus import wordnet as wn
import joblib
import nltk
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from pyspark.ml.feature import VectorAssembler, MinHashLSHModel,StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructField, IntegerType, StructType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, MinMaxScaler
from collections import Counter
# 设置 Spark 的日志级别
os.environ['PYSPARK_LOG_LEVEL'] = 'ERROR'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载数据集
df_last = pd.read_csv('dongman/222.csv')
# 创建Spark会话
spark = SparkSession.builder.appName("UserRecommendation").getOrCreate()
spark.conf.set("spark.sql.parquet.mergeSchema", "false")

schema = StructType([
    StructField("user_id", IntegerType(), True),
    StructField("anime_id", IntegerType(), True),
    StructField("rating", IntegerType(), True)
])

# 加载预处理过的老用户数据
preprocessed_data_path = "preprocessed_data.parquet"
preprocessed_data = spark.read.schema(schema).parquet(preprocessed_data_path)
class LSTMWithTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, l2_reg):
        super(LSTMWithTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_layer = nn.LSTMCell(input_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.l2_reg = l2_reg

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        h0 = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        hx = (h0, c0)
        for i in range(x.size(1)):
            hx = self.input_layer(x[:, i, :], hx)

        # 使用Transformer进行变换
        src = hx[0].unsqueeze(0)  # 将LSTM最后的hidden state作为Transformer的输入
        tgt = hx[0].unsqueeze(0)  # Transformer的目标输出等于输入
        out = self.transformer(src, tgt)

        out = self.fc(out.squeeze(0))
        return out

    def l2_regularization(self):
        l2_penalty = 0
        for param in self.parameters():
            l2_penalty += torch.norm(param, p=2)
        return self.l2_reg * l2_penalty

def encode_duration(duration_str):
    units = {'min': 60, 'sec': 1, 'hr': 3600}
    seconds = 0
    remaining_str = duration_str.strip()  # 去除字符串两端的空格
    count = 0

    for unit in ['hr', 'min', 'sec']:
        if unit in remaining_str:
            count += 1
            # 分割数字和单位
            parts = remaining_str.split(unit)
            if len(parts) > 1 and count <= 3:  # 确保分割后至少有两个部分且只取前三个单位
                num_str = parts[0].strip()  # 获取数字部分
                if num_str:  # 确保数字部分不为空
                    seconds += int(num_str) * units[unit]
                remaining_str = ' '.join(parts[1:]).strip()  # 更新剩余未处理的字符串
    if seconds == 0:
        return np.nan
    return seconds

# print("请按照提示输入数据")
# score_threshold = input("请输入想观看的动漫分数:如8.00\n")
# Aired=input("请输入上映日期：如1990-08-22或1990\n")
# Premiered_season=input("请输入上映季节：'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4, 'UNKNOWN': -1\n")
# Premiered_year=Aired
# Status=input("请输入上映状态：'Currently Airing': 1, 'Finished Airing': 2, 'Not yet aired': 3 'Unknown': -1\n")
# Source=input("请输入动漫起源：如'Manga', 'Light Novel', 'Visual Novel', 'Original', 等， 'Unknown': -1\n")
# Type=input("请输入动漫形式：如'Music', 'ONA', 'Movie', 'Special', 'OVA', 'TV', 'Unknown': -1\n")
# Duration_encoded1=input("请输入时长：如 1 hr 30 min sec 或 30 min per ep\n")
# Duration_encoded=encode_duration(Duration_encoded1)
# Episodes=input("请输入期望集数：\n")
# Members=input("请输入在平台上想看的成员数量：\n")
# ScoredBy=input("请输入评分人数：\n")
# Popularity=input("请输入人气排名：\n")
# Favorites=input("请输入被收藏数：\n")
# Studios=input("请输入动画工作室：如Sunrise，Madhouse\n")
# Rating=input("请输入动画年龄分集：如\n R - 17+ (violence & profanity): 1\n PG-13 - Teens 13 or older: 2\n "
#              "PG - Children: 3\n R+ - Mild Nudity:4\n G - All Ages: 5\n Rx - Hentai: 6\n")
# synopsis_encoded = input("请输入简介：\n")
# #Genres=input("请输入类型：如Action, Drama, Mystery, Supernatural，可多选\n")
# Producers=input("请输入制作公司：如Sunrise，Madhouse\n")

# input_features = pd.DataFrame({
#     'Score': [score_threshold],
#     'Aired': [Aired],
#     'Premiered_season': [Premiered_season],
#     'Premiered_year': [Premiered_year],
#     'Status': [Status],
#     'Source': [Source],
#     'Type': [Type],
#     'Duration_encoded': [Duration_encoded],
#     'Episodes': [Episodes],
#     'Members': [Members],
#     'Scored By': [ScoredBy],
#     'Popularity': [Popularity],
#     'Favorites': [Favorites],
#     'Studios':[Studios],
#     'Rating': [Rating],
#     'Synopsis': [synopsis_encoded],
#     #'Genres':[Genres],
#     'Producers':[Producers],
# })
# Score_value = input_features['Score'].values[0]
# Aired_value = input_features['Aired'].values[0]
# Premiered_season_value = input_features['Premiered_season'].values[0]
# Premiered_year_value = input_features['Premiered_year'].values[0]
# Status_value = input_features['Status'].values[0]
# Source_value = input_features['Source'].values[0]
# Type_value = input_features['Type'].values[0]
# Duration_encoded_value = input_features['Duration_encoded'].values[0]
# Episodes_value = input_features['Episodes'].values[0]
# Members_value = input_features['Members'].values[0]
# ScoredBy_value = input_features['Scored By'].values[0]
# Popularity_value = input_features['Popularity'].values[0]
# Favorites_value = input_features['Favorites'].values[0]
# Studios_value = input_features['Studios'].values[0]
# Rating_value = input_features['Rating'].values[0]
# Synopsis_encoded_value = input_features['Synopsis'].values[0]
# #Genres_value = input_features['Genres'].values[0]
# Producers_value = input_features['Producers'].values[0]
#
# data=['Score','Aired', 'Premiered_season', 'Premiered_year', 'Status', 'Source', 'Type','Duration',
#          'Episodes', 'Members', 'Scored By', 'Popularity', 'Favorites','Studios','Rating','Synopsis',
#        'Producers']
#
# data_list = [Score_value,Aired_value, Premiered_season_value, Premiered_year_value, Status_value,
#             Source_value, Type_value,Duration_encoded_value, Episodes_value, Members_value,
#              ScoredBy_value, Popularity_value, Favorites_value, Studios_value,Rating_value,
#              Synopsis_encoded_value,Producers_value]
#
# with open('dongman/预测name.csv', 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(data)
#
# with open('dongman/预测name.csv', 'a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(data_list)

file_path = 'userData.json'

# 读取并解析JSON文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
print(data)
# 将解析后的字典转换为DataFrame
df = pd.DataFrame(data)
df_labels = pd.read_csv('dongman/222.csv', encoding='utf-8')

# 处理'Aired'列缺失值
df['Aired'] = df['Aired'].astype(str)
df['Aired'] = df['Aired'].str.extract('(\d{4})').fillna(0).astype(int)

# 处理'Premiered'标签


# 处理'Status'标签
status_mapping = {'Currently Airing': 1, 'Finished Airing': 2, 'Not yet aired': 3}
df['Status'] = df['Status'].map(status_mapping)

# 处理'Source'标签
source_encoder = LabelEncoder()
df['Source_encoded'] = source_encoder.fit_transform(df['Source'])

# 处理'Genres'标签
genres_encoder = MultiLabelBinarizer()
genres_encoded = genres_encoder.fit_transform(
    df_labels['Genres'].str.split(',').apply(lambda x: [s.strip() for s in x] if isinstance(x, list) else []))
genres_encoded = pd.DataFrame(genres_encoded, columns=genres_encoder.classes_)

# 处理'Type'标签
type_encoder = LabelEncoder()
df['Type_encoded'] = type_encoder.fit_transform(df['Type'])

# 处理'synopsis'标签
nltk.download("wordnet", download_dir="nltk_data",quiet=True)

stop_words = set(stopwords.words('english'))


def augment_text_wordnet(text):
    words = word_tokenize(text)
    augmented_words = []
    for word in words:
        if word.lower() not in stop_words:
            synsets = wn.synsets(word)
            if synsets:
                synonyms = set()
                for synset in synsets:
                    for lemma in synset.lemmas():
                        synonyms.add(lemma.name().replace('_', ' '))
                if synonyms:
                    random_synonym = random.choice(list(synonyms))
                    augmented_words.append(random_synonym)
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)
    augmented_text = ' '.join(augmented_words)
    return augmented_text


df['Augmented_Synopsis_WordNet'] = df['Synopsis'].apply(
    lambda x: augment_text_wordnet(x) if isinstance(x, str) and len(x.strip()) > 0 else '')

# TF-IDF编码
synopsis_encoder = TfidfVectorizer()
valid_synopses = df[df['Augmented_Synopsis_WordNet'].astype(str).str.len() > 0]
synopsis_encoded = synopsis_encoder.fit_transform(valid_synopses['Augmented_Synopsis_WordNet'].apply(lambda x: str(x)))
synopsis_encoded = pd.DataFrame(synopsis_encoded.toarray(), columns=synopsis_encoder.get_feature_names_out())

# 处理'Rating'标签
rating_mapping = {'R - 17+ (violence & profanity)': 1, 'PG-13 - Teens 13 or older': 2, 'PG - Children': 3,
                  'R+ - Mild Nudity': 4, 'G - All Ages': 5, 'Rx - Hentai': 6}
df['Rating_encoded'] = df['Rating'].map(rating_mapping)

# 处理'Studios'标签
studios_encoder = LabelEncoder()
df['Studios'] = studios_encoder.fit_transform(df['Studios'])

# 处理'Producers'标签
producers_encoder = MultiLabelBinarizer()
producers_encoded = producers_encoder.fit_transform(
    df['Producers'].str.split(',').apply(lambda x: [s.strip() for s in x] if isinstance(x, list) else []))
producers_encoded = pd.DataFrame(producers_encoded, columns=producers_encoder.classes_)


# 编码'Duration'列
def encode_duration(duration_str):
    units = {'min': 60, 'sec': 1, 'hr': 3600}
    seconds = 0
    remaining_str = duration_str.strip()
    count = 0
    for unit in ['hr', 'min', 'sec']:
        if unit in remaining_str:
            count += 1
            parts = remaining_str.split(unit)
            if len(parts) > 1 and count <= 3:
                num_str = parts[0].strip()
                if num_str:
                    seconds += int(num_str) * units[unit]
                remaining_str = ' '.join(parts[1:]).strip()
    if seconds == 0:
        return np.nan
    return seconds


df['Duration_encoded'] = df['Duration'].apply(lambda x: encode_duration(str(x)) if str(x) != 'Unknown' else 0).astype(float)
df = df.fillna(-1)
# 合并处理后的数据
df_final = pd.concat([df[['ScoreThreshold','Aired', 'PremieredSeason', 'PremieredYear', 'Status', 'Source_encoded', 'Type_encoded','Duration_encoded',
                          'Episodes', 'Members', 'ScoreBy', 'Popularity', 'Favorites','Rating_encoded', 'Studios']],
                      synopsis_encoded,producers_encoded], axis=1)
ScoreThreshold = df['ScoreThreshold'].values
X = df_final
X = X.astype(np.float32)
input_tensor = torch.tensor(X.values, dtype=torch.float32)  # 将列表转换为张量


target_size = (19924, 55440)

# 计算每个维度需要填充的数量
pad_rows = target_size[0] - input_tensor.size(0)
pad_cols = target_size[1] - input_tensor.size(1)

# 在第一维度上填充行，在第二维度上填充列
padded_tensor = torch.nn.functional.pad(input_tensor, (0, pad_cols, 0, pad_rows), mode='constant', value=0)

padded_tensor = padded_tensor.to(device)



# # 读取原始 CSV 文件
# df = pd.read_csv('dongman/预测name.csv')
#
# # 删除第二行
# if len(df) > 1:
#     # 删除第二行
#     df.drop(index=1, inplace=True)
#
# # 保存修改后的 DataFrame 到新的文件
# df.to_csv('dongman/预测name.csv', index=False)

# 初始化循环神经网络模型
input_size = padded_tensor.shape[1]
hidden_size = 1024 # 隐含层大小
num_layers = 64 # LSTM层数
output_size = 22
model = LSTMWithTransformer(input_size, hidden_size, num_layers, output_size, l2_reg=0.001).to(device)
#model = nn.DataParallel(model, device_ids=[0, 1], output_device=1)
# 加载模型参数
checkpoint = torch.load('./lstm_trans_model.pth')
model.load_state_dict(checkpoint)

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查并设置使用GPU或CPU
model.to(device)
threshold = 0.9  # 设置阈值

# 在预测阶段使用加载的编码器进行标签解码
with torch.no_grad():
    padded_tensor = padded_tensor.to(device)
    predictions_tensor = model(padded_tensor)

predictions = predictions_tensor.cpu().numpy()

# 获取第一列前22个特征的索引并按照概率从大到小排序
top_indices = np.argsort(predictions[0])[::-1][:22]
top_indices = top_indices[top_indices < 22][:22]

# 创建一个字典来映射索引和对应的数据值
feature_dict = {
    0: "-1",
    1: "Action",
    2: "Adventure",
    3: "Avant Garde",
    4: "Award Winning",
    5: "Boys Love",
    6: "Comedy",
    7: "Drama",
    8: "Ecchi",
    9: "Erotica",
    10: "Fantasy",
    11: "Girls Love",
    12: "Gourmet",
    13: "Hentai",
    14: "Horror",
    15: "Mystery",
    16: "Romance",
    17: "Sci-Fi",
    18: "Slice of Life",
    19: "Sports",
    20: "Supernatural",
    21: "Suspense"
}
top_features = []
# 打印前三个概率最大的特征索引及其对应的概率值
for i in range(3):
    if i < len(top_indices):
        idx = top_indices[i]
        print(f"Top {i+1} - Feature Index: {idx}, Feature: {feature_dict.get(idx, 'Unknown')}, Probability: {predictions[0][idx]}")
        feature = feature_dict.get(idx, 'Unknown')
        probability = predictions[0][idx]
        top_features.append((feature, probability))
eve = predictions[0][top_indices[0]]-predictions[0][top_indices[21]]
sco2 = (predictions[0][top_indices[1]]-predictions[0][top_indices[21]])/eve
sco2*=10
sco3 = (predictions[0][top_indices[2]]-predictions[0][top_indices[21]])/eve
sco3*=10
score_list = [10,sco2,sco3]
id_list = []
for n in range(3):
    genre_filtered = df_labels[df_labels['Genres'].str.contains(top_features[n][0])]
    # 在这些数据中寻找Score标签值最高的数据
    max_score_row = genre_filtered.loc[genre_filtered['Score'].idxmax()]
    # 返回这条数据对应的anime_id标签
    anime_id = max_score_row['anime_id']
    id_list.append(anime_id)
    print(anime_id, max_score_row['Score'])
    print(anime_id,score_list[n])


def calculate_similarity(genres_list, predicted_categories):
    # 对genres_list中的元素计数
    genres_counter = Counter(genres_list)

    # 计算交集元素数量
    intersection = sum([genres_counter[category] for category in predicted_categories if category in genres_counter])

    # 计算并集元素数量
    union = sum(genres_counter.values()) + len(predicted_categories) - intersection

    # 使用Jaccard相似度计算相似度得分
    jaccard_similarity = intersection / float(union)

    return jaccard_similarity

similar_animes = {}
ScoreThreshold = float(ScoreThreshold)
for i, (feature, _) in enumerate(top_features):
    max_similarity_score = 0.0
    most_similar_anime = None

    for index, row in df_labels.iterrows():
        anime_genres = row['Genres'].split(',')
        similarity_score = float(calculate_similarity(anime_genres, [feature]))

        # 将 'Score' 字段转换为浮点数
        anime_score_float = float(row['Score'])

        if similarity_score > max_similarity_score and anime_score_float > ScoreThreshold:
            max_similarity_score = similarity_score
            most_similar_anime = row.copy()
            most_similar_anime['Similarity'] = max_similarity_score

    if most_similar_anime is not None:
        anime_id = most_similar_anime['anime_id']
        anime_score = most_similar_anime['Score']
        anime_similarity = most_similar_anime['Similarity']

        # print(f"\nFor the '{feature}' prediction category:")
        # print(f"Most Similar Anime ID: {anime_id}, Score: {anime_score}, Similarity: {anime_similarity}")
        similar_animes[feature] = most_similar_anime
anime_id_list = []
ScoreThreshold = str(ScoreThreshold)
# 假设 `similar_animes` 是储存了每个预测类别最相似动漫信息的字典
for category, anime_info in similar_animes.items():
    anime_id = anime_info['anime_id']
    anime_id_list.append(anime_id)
    print(f"The most similar anime for category '{category}' has an ID: {anime_id}")

input = [['3581', anime_id_list[0], int(score_list[0])],
        ['3581', anime_id_list[1], int(score_list[1])],
        ['3581', anime_id_list[2],int(score_list[2])]]
#print(input)
    # 对预测值进行反向转换
    #predicted_name = name_encoder.inverse_transform([int(predictions_np[0])])

#print(predicted_name)

data = input
# data = [['3580', '43608', '9'],
#         ['3580', '5114', '9'],
#         ['3580', '45576', '8']]

columns = ['user_id', 'anime_id', 'rating']

df_spark = pd.DataFrame(data, columns=columns)
file_path_spark = 'newuser.csv'
df_spark.to_csv(file_path_spark, index=False)

# 读取数据集并设置分区

new_user_data = spark.read.csv(file_path_spark, header=True, inferSchema=True)

new_user_data = new_user_data.withColumn('user_id', col('user_id').cast(IntegerType()))


#将用户-动漫关联矩阵转换为特征向量
assembler = VectorAssembler(inputCols=new_user_data.columns[1:], outputCol="features")
output = assembler.transform(new_user_data)
#创建标准化器
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
scalerModel = scaler.fit(output)
output = scalerModel.transform(output)
output = new_user_data
#print(output.columns)
#print(preprocessed_data.columns)
# 合并预处理过的老用户数据和新用户数据
# 这里假设新用户数据已经通过相同的预处理步骤转换为特征向量并标准化
combined_data = preprocessed_data.unionByName(output, allowMissingColumns=False)

assembler = VectorAssembler(inputCols=combined_data.columns[1:], outputCol="features")
output = assembler.transform(combined_data)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
scalerModel = scaler.fit(output)
output = scalerModel.transform(output)
# 加载已训练的模型
model_path = "minhash_lsh_model"
model = MinHashLSHModel.load(model_path)

# 定义获取推荐的函数
def get_recommendations_for_new_user(user_id, num_recommendations=20):
    user_features = output.filter(col("user_id") == user_id).select("scaledFeatures").collect()[0]["scaledFeatures"]
    new_user_features = Vectors.dense(user_features.toArray())
    #print(new_user_features)
    similar_items = model.approxNearestNeighbors(output, new_user_features, num_recommendations)
    recommendations = similar_items.select("user_id").rdd.flatMap(lambda x: x).collect()
    return recommendations


# 获取新用户的推荐
user_id = '3581'
recommendations = get_recommendations_for_new_user(user_id)
#print("推荐项：", recommendations)

# 五个已知的anime_id值
known_anime_ids = [recommendations[0], recommendations[1], recommendations[2], recommendations[3],
                   recommendations[4], recommendations[5], recommendations[6], recommendations[7],
                   recommendations[8],recommendations[9],recommendations[10],recommendations[11],
                   recommendations[12],recommendations[13],recommendations[14],recommendations[15],
                   recommendations[16],recommendations[17],recommendations[18],recommendations[19]]



# 定义你想要选取的列名
desired_columns = ['Name', 'Other name', 'Genres', 'Score', 'Image URL']

ScoreThreshold = float(ScoreThreshold)
# 筛选出对应anime_id且评分高于ScoreThreshold的数据行
filtered_rows = df_last[(df_last['anime_id'].isin(known_anime_ids)) & (df_last['Score'] > ScoreThreshold)][desired_columns]

# 将筛选结果保存到新的DataFrame中
selected_data = filtered_rows.reset_index(drop=True)

# 因为我们已经筛选出了评分高于门槛的行，这里不再需要排序，直接取前五个即可
selected_data = selected_data.head(5)

last = []
for _, row in selected_data.iterrows():
    anime_dict = dict(row)
    # 修改字典的键名
    anime_dict['OtherName'] = anime_dict.pop('Other name')
    anime_dict['ImageURL'] = anime_dict.pop('Image URL')
    last.append(anime_dict)

# 将结果写入到 JSON 文件中
json_data = json.dumps(last, ensure_ascii=False)
with open('resultData.json', 'w') as json_file:
    json_file.write(json_data)

# 停止Spark会话
spark.stop()
print("ok")