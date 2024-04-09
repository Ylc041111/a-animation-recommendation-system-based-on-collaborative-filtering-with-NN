import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, MinMaxScaler
import nltk
from nltk.corpus import wordnet as wn
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset, DataLoader

# GPU设备检测
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 数据预处理
df = pd.read_csv('dongman/222.csv')  # 读取数据文件路径

# 处理'Aired'列缺失值
df['Aired'] = df['Aired'].str.extract('(\d{4})').fillna(0).astype(int)

# 处理'Premiered'标签
season_mapping = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4, 'UNKNOWN': 0}
df['Premiered'] = df['Premiered'].str.split(' ')
df['Premiered'] = df['Premiered'].apply(lambda x: ([season_mapping.get(x[0], 0), int(x[1])] if len(x) >= 2 else [0, 0])
                                        )
df[['Premiered_season', 'Premiered_year']] = pd.DataFrame(df['Premiered'].to_list(), index=df.index)
df.drop('Premiered', axis=1, inplace=True)

# 处理'Status'标签
status_mapping = {'Currently Airing': 1, 'Finished Airing': 2, 'Not yet aired': 3}
df['Status'] = df['Status'].map(status_mapping)

# 处理'Source'标签
source_encoder = LabelEncoder()
df['Source_encoded'] = source_encoder.fit_transform(df['Source'])

# 处理'Genres'标签
genres_encoder = MultiLabelBinarizer()
genres_encoded = genres_encoder.fit_transform(
    df['Genres'].str.split(',').apply(lambda x: [s.strip() for s in x] if isinstance(x, list) else []))
genres_encoded = pd.DataFrame(genres_encoded, columns=genres_encoder.classes_)

# 处理'Type'标签
type_encoder = LabelEncoder()
df['Type_encoded'] = type_encoder.fit_transform(df['Type'])

# 处理'synopsis'标签
nltk.download("wordnet", download_dir="nltk_data")

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


df['Duration_encoded'] = df['Duration'].apply(lambda x: encode_duration(x) if x != 'Unknown' else 0).astype(float)
df = df.fillna(-1)

df_features = pd.concat([df[['Score','Aired', 'Premiered_season', 'Premiered_year', 'Status', 'Source_encoded', 'Type_encoded',
                        'Duration_encoded' , 'Episodes', 'Members', 'Scored By', 'Popularity',
                          'Favorites', 'Studios', 'Rating_encoded']], synopsis_encoded, producers_encoded],
                     axis=1)
df_labels = genres_encoded
del(df)
del(synopsis_encoded)
del(producers_encoded)
x = df_features
y = df_labels
num_classes = y.nunique()
print("Number of classes in y:", num_classes)

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 将训练集和测试集转换为PyTorch Tensor

#X_train_tensor = torch.tensor(X_train.values, dtype=torch.long).to(device)

#X_test_tensor = torch.tensor(X_test.values, dtype=torch.long).to(device)
# 确保y_train和y_test已转换为二进制矩阵形式

#y_train_tensor = torch.tensor(y_train.values.reshape(-1, 109582), dtype=torch.long).to(device)

#y_test_tensor = torch.tensor(y_test.values.reshape(-1, 109582), dtype=torch.long).to(device)





# 定义LSTM模型
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

class CustomDataset(Dataset):
    def __init__(self, X_data, y_data):
        # 将 DataFrame 转换为 numpy 数组
        self.X_data = torch.tensor(X_data.values.astype(np.float32))
        self.y_data = torch.tensor(y_data.values.astype(np.float32))

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]
# 辅助函数
# 计算准确率
def calculate_accuracy(outputs, targets):
    predictions = (outputs > 0.5).float()  # 获取预测结果
    correct = (predictions == targets).sum(dim=1).float()  # 计算每条样本的正确类别数
    total_classes = targets.size(1)
    accuracy = correct.sum().item() / (total_classes * targets.size(0))  # 平均准确率
    return accuracy

# 计算召回率
def calculate_recall(outputs, targets, average='micro'):
    recall = recall_score(targets, outputs, average=average)
    return recall


# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Shape of X_train:", X_train.shape)
# 初始化模型参数
input_size = X_train.shape[1]
hidden_size = 1024
num_layers = 64
output_size = 22
l2_reg = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# 训练函数
# 添加召回率记录
# 训练函数
def train_model(model, optimizer, criterion, train_loader, test_loader, num_epochs=100):
    train_losses = []
    train_accuracies = []
    train_recalls = []
    test_losses = []
    test_accuracies = []
    test_recalls = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        epoch_train_accuracies = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            y_target_resized = target[:, :outputs.size(1)]
            loss = criterion(outputs, y_target_resized)
            l2_loss = model.l2_regularization()
            total_loss = loss + l2_loss
            total_loss.backward()
            optimizer.step()
            accuracy = calculate_accuracy(outputs, target)
            epoch_train_losses.append(total_loss.item())
            epoch_train_accuracies.append(accuracy)

        # 在每个epoch结束时，计算整个训练集上的召回率
        train_recall = evaluate_model(model, train_loader, criterion)[1]
        train_losses.extend(epoch_train_losses)
        train_accuracies.extend(epoch_train_accuracies)
        train_recalls.append(train_recall)
        train_loss_avg = sum(epoch_train_losses) / len(epoch_train_losses)
        train_accuracy_avg = sum(epoch_train_accuracies) / len(epoch_train_accuracies)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss_avg:.4f}, Train Accuracy: {train_accuracy_avg:.4f}, Train Recall: {train_recall:.4f}')

        # 在每个epoch结束时，计算整个测试集上的损失、准确率和召回率
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        test_recall = evaluate_model(model, test_loader,criterion)[1]
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_recalls.append(test_recall)
        test_loss_avg = sum(test_losses) / len(test_losses)
        test_accuracy_avg = sum(test_accuracies) / len(test_accuracies)
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss_avg:.4f}, Test Accuracy: {test_accuracy_avg:.4f}, Test Recall: {test_recall:.4f}')

    return train_losses, train_accuracies, train_recalls, test_losses, test_accuracies, test_recalls


def evaluate_model(model, loader, criterion):
    model.eval()
    all_losses = []
    all_accuracies = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            y_target_resized = target[:, :outputs.size(1)]
            loss = criterion(outputs, y_target_resized)
            accuracy = calculate_accuracy(outputs, target)
            all_losses.append(loss.item())
            all_accuracies.append(accuracy)

    return sum(all_losses)/len(all_losses), sum(all_accuracies)/len(all_accuracies)
# 创建数据加载器
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
batch_size = 1280
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Preprocessing completed successfully.")
# 初始化模型、优化器和损失函数
model = LSTMWithTransformer(input_size, hidden_size, num_layers, output_size, l2_reg).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()
losses = []
accuracies = []
recalls = []
# 训练模型
train_losses, train_accuracies, train_recalls, test_losses, test_accuracies, test_recalls = train_model(model, optimizer, criterion, train_loader, test_loader, num_epochs=100)


# 保存最后一次训练得到的模型状态
torch.save(model.state_dict(), 'last_prediction_model.pth')

# 绘制损失、准确率和召回率曲线
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_recalls, label='Train Recall')
plt.plot(test_recalls, label='Test Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Training and Testing Recall')
plt.legend()

plt.show()