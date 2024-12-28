import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

from model import FNN, CNN, ResNetModel
from prediction import predictor, quarterfinal_predictor

# 讀取訓練與測試數據
train_file_path = 'train_lol_cleaned.csv'
test_file_path = 'test_lol_cleaned.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

target_list = ['A_wins', "A_firstTowerKill", "A_firstBlood",
                'B_firstTowerKill', "B_firstBlood"]

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
y_train_firstBlood_tensor = torch.tensor(y_train["A_firstBlood"].values, dtype=torch.float32).unsqueeze(1)
y_train_firstTower_tensor = torch.tensor(y_train["A_firstTowerKill"].values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
y_test_wins_tensor = torch.tensor(y_test["A_wins"].values, dtype=torch.float32).unsqueeze(1)
y_test_firstBlood_tensor = torch.tensor(y_test["A_firstBlood"].values, dtype=torch.float32).unsqueeze(1)
y_test_firstTower_tensor = torch.tensor(y_test["A_firstTowerKill"].values, dtype=torch.float32).unsqueeze(1)

# 初始化模型、損失函數和優化器
input_size = X_train_tensor.shape[1]
model = ResNetModel(input_size,3)
criterion = nn.BCELoss()  # 二元交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 訓練模型
epochs = 30
batch_size = 64

class MultiTargetDataset(Dataset):
    def __init__(self, X, y_wins, y_firstBlood, y_firstTower):
        self.X = X
        self.y_wins = y_wins
        self.y_firstBlood = y_firstBlood
        self.y_firstTower = y_firstTower

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_wins[idx], self.y_firstBlood[idx], self.y_firstTower[idx]

train_dataset = MultiTargetDataset(X_train_tensor, y_train_wins_tensor, y_train_firstBlood_tensor, y_train_firstBlood_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()

    epoch_loss = 0
    for batch in train_loader:
        X_batch, y_wins_batch, y_firstBlood_batch, y_firstTower_batch = batch
        optimizer.zero_grad()
        wins, firstBlood, firstTower = model(X_batch)
        # 計算損失
        loss_wins = criterion(wins, y_wins_batch)
        loss_firstBlood = criterion(firstBlood, y_firstBlood_batch)
        loss_firstTower = criterion(firstTower, y_firstTower_batch)
        loss = loss_wins + loss_firstBlood + loss_firstTower
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(X_train_tensor):.4f}")

# 測試模型
model.eval()
with torch.no_grad():
    wins, firstBlood, firstTower = model(X_test_tensor)
    A_wins = wins
    A_firstBloodKill = firstBlood
    A_firstTowerKill = firstTower
    B_firstBloodKill = 1 - A_firstBloodKill
    B_firstTowerKill = 1 - A_firstTowerKill

    predictions = {
        "A_wins": A_wins,
        "A_firstBlood": A_firstBloodKill,
        "B_firstBlood": B_firstBloodKill,
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

# 去掉目標欄位，並標準化數據
X_test_features = test_data.drop(columns=target_list + ["game_date", "A_teamname", "B_teamname"])
X_test_scaled, _ = preprocess_with_scaler(X_test_features, scaler=scaler)

# 確保傳遞完整的測試數據
test_data_scaled = test_data.copy()
test_data_scaled[X_test_features.columns] = X_test_scaled[X_test_features.columns]
#predictor(test_data_scaled, model, model_name="FNN")
#predictor(test_data_scaled, model, model_name="CNN")
predictor(test_data_scaled, model, model_name="ResNetModel")

# 呼叫 predictor 函數，傳遞完整的測試數據
def draw_tournament_hierarchy(quarterfinal_matches, quarter_winners,
                              semifinal_winners, final_winner, tournament_name):
    """
    繪製錦標賽圖表
    :param matches: 比賽的列表，每場比賽是 (隊伍A, 隊伍B)
    :param winners: 對應的勝者列表，按比賽順序排列
    :param stage: 比賽階段名稱 (如 "Quarterfinals", "Semifinals", "Finals")
    """
    G = nx.DiGraph()

    # Ensure inputs are lists or empty lists
    quarter_winners = quarter_winners or []
    semifinal_winners = semifinal_winners or []
    final_winner = final_winner or "TBD"

    # Adding quarterfinal matches
    for i, match in enumerate(quarterfinal_matches):
        G.add_node(f"Q{i+1}A", label=match[0])
        G.add_node(f"Q{i+1}B", label=match[1])
        winner_label = quarter_winners[i] if i < len(quarter_winners) else "TBD"
        G.add_node(f"Q{i+1}W", label=winner_label)
        G.add_edge(f"Q{i+1}A", f"Q{i+1}W")
        G.add_edge(f"Q{i+1}B", f"Q{i+1}W")

    # Adding semifinal matches
    for i in range(2):
        semifinal_winner_label = semifinal_winners[i] if i < len(semifinal_winners) else "TBD"
        G.add_node(f"S{i+1}W", label=semifinal_winner_label)
        G.add_edge(f"Q{2*i+1}W", f"S{i+1}W")
        G.add_edge(f"Q{2*i+2}W", f"S{i+1}W")
    # Adding final match
    final_winner_label = final_winner if final_winner else "TBD"
    G.add_node("F1W", label=final_winner_label)
    G.add_edge("S1W", "F1W")
    G.add_edge("S2W", "F1W")

    pos = {f"Q{i+1}A": (0, -i * 2) for i in range(4)}
    pos.update({f"Q{i+1}B": (0, -i * 2 - 1) for i in range(4)})
    pos.update({f"Q{i+1}W": (1, -i * 2 - 0.5) for i in range(4)})

    pos.update({f"S{i+1}W": (2, -i * 4 - 1) for i in range(2)})
    pos.update({"F1W": (3, -2)})

    labels = nx.get_node_attributes(G, "label")
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_color="black")

    plt.title("Tournament Hierarchy")
    plt.axis("off")
    plt.gcf().canvas.manager.set_window_title(tournament_name)
    plt.show()

#predictor(test_data_scaled, model)
def worlds_2024():
    # 定義八強比賽列表
    quarterfinal_matches = [
        ("LNG", "WB"),
        ("HLE", "BLG"),
        ("TES", "T1"),
        ("GEN", "FLY")
    ]
    print(f"八強比賽列表：{quarterfinal_matches}")
    (quarter_match_winner,
     quarter_first_blood,
     quarter_first_tower) = quarterfinal_predictor(test_data_scaled,
                                                   model, 
                                                   quarterfinal_matches,
                                                   4,
                                                   "2024-10-17T00:00:00.000Z")
    print(f"八強比賽預測結果：{quarter_match_winner}")
    #draw_tournament_hierarchy(quarterfinal_matches, quarter_match_winner, [], "")

    semifinal_matches = [
        (quarter_match_winner[0], quarter_match_winner[1]),
        (quarter_match_winner[2], quarter_match_winner[3])
    ]
    (semi_match_winner,
     semi_first_blood,
     semi_first_tower) = quarterfinal_predictor(test_data_scaled,
                                                model,
                                                semifinal_matches,
                                                7,
                                                "2024-10-26T00:00:00.000Z")
    print(f"四強比賽預測結果：{semi_match_winner}")
    #draw_tournament_hierarchy(quarterfinal_matches, quarter_match_winner, semi_match_winner, "")

    final_matches = [
        (semi_match_winner[0], semi_match_winner[1])
    ]
    (final_match_winner,
     final_first_blood,
     final_first_tower) = quarterfinal_predictor(test_data_scaled,
                                                 model,
                                                 final_matches,
                                                 10,
                                                 "2024-11-02T00:00:00.000Z")
    print(f"冠軍：{final_match_winner[0]}")
    draw_tournament_hierarchy(quarterfinal_matches, quarter_match_winner,
                              semi_match_winner, final_match_winner[0], "LOL Worlds 2024")

def worlds_2023():
    # 定義八強比賽列表
    quarterfinal_matches = [
        ("GEN", "BLG"),
        ("NRG", "WB"),
        ("JDG", "KT"),
        ("LNG", "T1")
    ]
    print(f"八強比賽列表：{quarterfinal_matches}")
    (quarter_match_winner,
     quarter_first_blood,
     quarter_first_tower) = quarterfinal_predictor(test_data_scaled,
                                                   model, 
                                                   quarterfinal_matches,
                                                   4,
                                                   "2023-11-02T00:00:00.000Z")
    print(f"八強比賽預測結果：{quarter_match_winner}")
    #draw_tournament_hierarchy(quarterfinal_matches, quarter_match_winner, [], "")

    semifinal_matches = [
        (quarter_match_winner[0], quarter_match_winner[1]),
        (quarter_match_winner[2], quarter_match_winner[3])
    ]
    (semi_match_winner,
     semi_first_blood,
     semi_first_tower) = quarterfinal_predictor(test_data_scaled,
                                                model,
                                                semifinal_matches,
                                                7,
                                                "2023-11-11T00:00:00.000Z")
    print(f"四強比賽預測結果：{semi_match_winner}")
    #draw_tournament_hierarchy(quarterfinal_matches, quarter_match_winner, semi_match_winner, "")

    final_matches = [
        (semi_match_winner[0], semi_match_winner[1])
    ]
    (final_match_winner,
     final_first_blood,
     final_first_tower) = quarterfinal_predictor(test_data_scaled,
                                                 model,
                                                 final_matches,
                                                 10,
                                                 "2023-11-19T00:00:00.000Z")
    print(f"冠軍：{final_match_winner[0]}")
    draw_tournament_hierarchy(quarterfinal_matches, quarter_match_winner,
                              semi_match_winner, final_match_winner[0], "LOL Worlds 2023")

worlds_2024()
worlds_2023()