import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


from model import FCNN, CNN

# 讀取訓練與測試數據
train_file_path = 'train_lol_cleaned.csv'
test_file_path = 'test_lol_cleaned.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

target_list = ['A_wins', "A_firstInhibitorKill", "A_firstTowerKill", 'B_firstInhibitorKill', 'B_firstTowerKill']
#target = ['A_wins']

# 分離特徵與目標變量
X_train = train_data.drop(columns=target_list)
y_train = train_data[target_list]

X_test = test_data.drop(columns=target_list)
y_test = test_data[target_list]

def preprocess_with_scaler(df, scaler=None):
    """
    標準化數值特徵並保留非數值特徵。
    :param df: 原始 DataFrame
    :param scaler: 標準化的 Scaler（如 StandardScaler），如果為 None，將自動創建
    :return: 經過標準化的 DataFrame 和 Scaler
    """
    numeric = df.select_dtypes(include=['number'])
    non_numeric = df.select_dtypes(exclude=['number'])
    
    if scaler is None:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(numeric)
    else:
        numeric_scaled = scaler.transform(numeric)
    
    # 處理因數據全為零導致的 NaN
    numeric_scaled = pd.DataFrame(numeric_scaled, columns=numeric.columns, index=df.index)
    numeric_scaled.fillna(0, inplace=True)
    
    combined_df = pd.concat([non_numeric.reset_index(drop=True), numeric_scaled.reset_index(drop=True)], axis=1)
    return combined_df, scaler


# 標準化特徵
scaler = StandardScaler()
X_train = X_train.drop(columns=["game_date", "A_teamname", "B_teamname"])
X_test = X_test.drop(columns=["game_date", "A_teamname", "B_teamname"])
X_train_scaled, scaler = preprocess_with_scaler(X_train)
X_test_scaled, _ = preprocess_with_scaler(X_test, scaler=scaler)

print(X_train_scaled.isna().sum())
print(y_train.isna().sum())

print(X_train_scaled.head(5))
print(X_test_scaled.head(5))

# 轉換為 PyTorch 張量
X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
y_train_wins_tensor = torch.tensor(y_train["A_wins"].values, dtype=torch.float32).unsqueeze(1)
y_train_firstInhibitor_tensor = torch.tensor(y_train["A_firstInhibitorKill"].values, dtype=torch.float32).unsqueeze(1)
y_train_firstTower_tensor = torch.tensor(y_train["A_firstTowerKill"].values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_test_wins_tensor = torch.tensor(y_test["A_wins"].values, dtype=torch.float32).unsqueeze(1)
y_test_firstInhibitor_tensor = torch.tensor(y_test["A_firstInhibitorKill"].values, dtype=torch.float32).unsqueeze(1)
y_test_firstTower_tensor = torch.tensor(y_test["A_firstTowerKill"].values, dtype=torch.float32).unsqueeze(1)

# 初始化模型、損失函數和優化器
input_size = X_train_tensor.shape[1]
model = CNN(input_size)
criterion = nn.BCELoss()  # 二元交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 訓練模型
epochs = 30
batch_size = 32

class MultiTargetDataset(Dataset):
    def __init__(self, X, y_wins, y_firstInhibitor, y_firstTower):
        self.X = X
        self.y_wins = y_wins
        self.y_firstInhibitor = y_firstInhibitor
        self.y_firstTower = y_firstTower

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_wins[idx], self.y_firstInhibitor[idx], self.y_firstTower[idx]

train_dataset = MultiTargetDataset(X_train_tensor, y_train_wins_tensor, y_train_firstInhibitor_tensor, y_train_firstInhibitor_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()

    epoch_loss = 0
    for batch in train_loader:
        X_batch, y_wins_batch, y_firstInhibitor_batch, y_firstTower_batch = batch
        optimizer.zero_grad()
        wins, firstInhibitor, firstTower = model(X_batch)
        # 計算損失
        loss_wins = criterion(wins, y_wins_batch)
        loss_firstInhibitor = criterion(firstInhibitor, y_firstInhibitor_batch)
        loss_firstTower = criterion(firstTower, y_firstTower_batch)
        loss = loss_wins + loss_firstInhibitor + loss_firstTower
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(X_train_tensor):.4f}")

# 測試模型
model.eval()
with torch.no_grad():
    wins, firstInhibitor, firstTower = model(X_test_tensor)
    A_firstInhibitorKill = (firstInhibitor > 0.5).float()
    A_firstTowerKill = (firstTower > 0.5).float()
    B_firstInhibitorKill = 1 - A_firstInhibitorKill
    B_firstTowerKill = 1 - A_firstTowerKill

    predictions = {
        "A_wins": (wins > 0.5).float(),
        "A_firstInhibitorKill": A_firstInhibitorKill,
        "B_firstInhibitorKill": B_firstInhibitorKill,
        "A_firstTowerKill": A_firstTowerKill,
        "B_firstTowerKill": B_firstTowerKill
    }

    for target in target_list:
        print(f"{target} roc_auc_score: {roc_auc_score(y_test[target], predictions[target])}")
    
    # draw all target roc_auc_score pyplot in one figure
    fig = plt.figure(figsize=(15, 10))
    for idx, target in enumerate(target_list):
        fpr, tpr, _ = roc_curve(y_test[target], predictions[target])
        ax = fig.add_subplot(2, 3, idx+1)
        ax.plot(fpr, tpr, label=f"{target} ROC curve (area = {roc_auc_score(y_test[target], predictions[target]):.2f})")
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title(f'{target} ROC curve')
        ax.legend(loc='best')
    plt.tight_layout()
    plt.show()