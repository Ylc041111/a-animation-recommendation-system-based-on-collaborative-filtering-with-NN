import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 读取数据
df = pd.read_csv('dongman/222.csv')

# 处理缺失值
df['Aired'] = df['Aired'].str.extract('(\d{4})').fillna(0)
df['Aired'] = df['Aired'].astype(int)

# 处理Premiered标签
season_mapping = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4, 'UNKNOWN': 0}
df['Premiered'] = df['Premiered'].str.split(' ')
df['Premiered'] = df['Premiered'].apply(lambda x: ([season_mapping.get(x[0], 0), int(x[1])] if len(x) >= 2 else [0, 0]) )
df[['Premiered_season', 'Premiered_year']] = pd.DataFrame(df['Premiered'].to_list(), index=df.index)
df.drop('Premiered', axis=1, inplace=True)

# 处理Status标签
status_mapping = {'Currently Airing': 1, 'Finished Airing': 2, 'Not yet aired': 3}
df['Status'] = df['Status'].map(status_mapping)

# 处理Source标签
source_encoder = LabelEncoder()
df['Source'] = source_encoder.fit_transform(df['Source'])

# 处理Genres标签
genres_encoder = MultiLabelBinarizer()
genres_encoded = genres_encoder.fit_transform(df['Genres'].str.split(',').apply(lambda x: [s.strip() for s in x] if isinstance(x, list) else []))
genres_encoded = pd.DataFrame(genres_encoded, columns=genres_encoder.classes_)

# 处理Type标签
type_encoder = LabelEncoder()
df['Type'] = type_encoder.fit_transform(df['Type'])

# 处理synopsis标签
synopsis_encoder = TfidfVectorizer(max_features=500)  # 选择500个最重要的特征
synopsis_encoded = synopsis_encoder.fit_transform(df['Synopsis'].values.astype('U'))  # 转换文本数据为稀疏矩阵
synopsis_encoded = pd.DataFrame(synopsis_encoded.toarray(), columns=synopsis_encoder.get_feature_names_out())  # 转换为DataFrame

def encode_duration(duration_str):
    units = {'min': 60, 'sec': 1, 'hr': 3600}
    seconds = 0
    remaining_str = duration_str.strip() # 去除字符串两端的空格
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

#在预处理阶段执行替换和转换操作
df['Duration_encoded'] = df['Duration'].apply(lambda x: encode_duration(x) if x != 'Unknown' else 0).astype(float)
df = df.fillna(0)

# 合并处理后的数据
df_final = pd.concat([df[['Aired', 'Premiered_season', 'Premiered_year', 'Status', 'Source', 'Type', 'Duration_encoded',
                          'Episodes', 'Members', 'Scored By', 'Popularity', 'Favorites']], synopsis_encoded,genres_encoded], axis=1)

# 建立特征集和标签集
X = df_final
y = df['Score']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建rf模型
model = RandomForestRegressor(n_estimators=100)

# 拟合模型
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)

# 评估模型
correlation = np.corrcoef(y_test, predictions)[0, 1]
print(f"Correlation: {correlation}")

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

r_squared = r2_score(y_test, predictions)
print(f"R-squared: {r_squared}")

importance_scores = model.feature_importances_
feature_names = X_train.columns
for feature, importance in zip(feature_names, importance_scores):
    print(f"{feature}: {importance}")
