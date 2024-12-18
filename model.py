import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("每個模型必須實現 forward 方法")

class FCNN(BaseModel):
    def __init__(self, input_size):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output_wins = nn.Linear(32, 1)  # 單一輸出
        self.output_firstInhibitorKill = nn.Linear(32, 1)
        self.output_firstTowerKill = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()  # 適合二分類
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        # 分別輸出三個結果
        wins = self.output_wins(x)
        wins = self.sigmoid(wins)
        wins = torch.nan_to_num(wins, nan=0.0)  # 防止 NaN
        
        firstInhibitor = self.output_firstInhibitorKill(x)
        firstInhibitor = self.sigmoid(firstInhibitor)
        
        firstTower = self.output_firstTowerKill(x)
        firstTower = self.sigmoid(firstTower)
        
        return wins, firstInhibitor, firstTower

class CNN(BaseModel):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output_wins = nn.Linear(32, 1)  # 單一輸出
        self.output_firstInhibitorKill = nn.Linear(32, 1)
        self.output_firstTowerKill = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()  # 適合二分類
        self.softmax = nn.Softmax(dim=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一個維度以適應 Conv1d
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # 分別輸出三個結果
        wins = self.output_wins(x)
        wins = self.sigmoid(wins)  # 確保輸出在 [0, 1] 範圍內
        
        firstInhibitor = self.output_firstInhibitorKill(x)
        firstInhibitor = self.sigmoid(firstInhibitor)  # 輸出概率值
        
        firstTower = self.output_firstTowerKill(x)
        firstTower = self.sigmoid(firstTower)  # 輸出概率值
        
        return wins, firstInhibitor, firstTower