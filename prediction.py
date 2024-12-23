import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def convert_to_teamdata(df):
    general_attr = ["game_date"]
    
    # 這裡如果已經改成a和b的話判斷要做修改
    A_attr = [col for col in df.columns if "A_" in col and col not in ["A_wins"]]
    B_attr = [col for col in df.columns if "B_" in col]
    neutral_attr = [col.replace("B_", "") for col in B_attr]

    df["game_date"] = pd.to_datetime(df["game_date"])
    general = df[general_attr]
    
    A = df[A_attr]
    B = df[B_attr]

    # 將 blue 和 red 的列名統一
    A.columns = neutral_attr
    B.columns = neutral_attr

    # 合併 blue 和 red DataFrame
    team_data = pd.concat([A, B], axis=0, ignore_index=True)
    team_data["game_date"] = pd.concat([general["game_date"], general["game_date"]], axis=0, ignore_index=True)
    team_data = team_data.sort_values("game_date", ascending=False)

    return team_data

def find_most_recent_games(df, team_name, n_games, compete_date=None):
    """
    Find the most recent games before compete date for a given team
    """
    if compete_date is not None:
        df = df[df["game_date"] < compete_date]
    team_games = df[(df["teamname"] == team_name)]
    return team_games.head(n_games)

def produce_match_list(team_data_1, team_data_2):
    """
    Based on the data of two teams, combine each other to produce new match list
    """
    match_list_left = []
    match_list_right = []

    for i, game_1 in team_data_1.iterrows():
        for j, game_2 in team_data_2.iterrows():
            # set game_1 columns to all have "a_" prefix, and game_2 columns to all have "b_" prefix
            game_1 = game_1.rename(lambda x: "a_" + x if not x.startswith("a_") else x)
            game_2 = game_2.rename(lambda x: "b_" + x if not x.startswith("b_") else x)
            # combine them into a single row
            match = pd.concat([game_1, game_2])
            match_list_left.append(match)

    match_df_left = pd.DataFrame(match_list_left)
    match_df_left.drop(columns=["a_teamname", "b_teamname", "a_game_date", "b_game_date", "a_firstInhibitorKill", "b_firstInhibitorKill", "a_firstTowerKill", "b_firstTowerKill"], inplace=True)
 
    for i, game_1 in team_data_2.iterrows():
        for j, game_2 in team_data_1.iterrows():
            # set game_1 columns to all have "a_" prefix, and game_2 columns to all have "b_" prefix
            game_1 = game_1.rename(lambda x: "a_" + x if not x.startswith("a_") else x)
            game_2 = game_2.rename(lambda x: "b_" + x if not x.startswith("b_") else x)
            # combine them into a single row
            match = pd.concat([game_1, game_2])
            match_list_right.append(match)
    
    match_df_right = pd.DataFrame(match_list_right)
    match_df_right.drop(columns=["a_teamname", "b_teamname", "a_game_date", "b_game_date", "a_firstInhibitorKill", "b_firstInhibitorKill", "a_firstTowerKill", "b_firstTowerKill"], inplace=True)

    '''
    偶數場隨機刪一場
    '''
    '''# 隨機刪除一個 row
    if len(match_df) > 1:
        random_index = np.random.choice(match_df.index)
        match_df = match_df.drop(random_index).reset_index(drop=True)
        
        print(f"已隨機刪除 row 索引: {random_index}")
        #match_list.to_csv("match_list.csv", index=False)'''
    
    print("Left Match Data NaN Check:", match_df_left.isnull().sum().sum())
    print("Right Match Data NaN Check:", match_df_right.isnull().sum().sum())

    return match_df_left, match_df_right

def predictor(model):
    test_data = pd.read_csv("test_lol_cleaned.csv")
    print("Initial Test Data:")
    print(test_data.info())
    print(test_data.isnull().sum())
    print(test_data.head(3))

    team_data = convert_to_teamdata(test_data)
    #team_data = team_data.loc[:, ~team_data.columns.duplicated()]
    print("Converted Team Data:")
    print(team_data.info())
    print(team_data.isnull().sum())
    print(team_data.head(3))

    # 定義目標
    targets = ["A_wins", "A_firstInhibitorKill", "A_firstTowerKill"]
    actual_results = {target: [] for target in targets}
    predicted_results = {target: [] for target in targets}

    for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing CSV file"):
        A_teamname = row["A_teamname"]
        B_teamname = row["B_teamname"]
        game_date = row["game_date"]
        
        team_1_game_data = find_most_recent_games(team_data, A_teamname, 3, game_date)
        team_2_game_data = find_most_recent_games(team_data, B_teamname, 3, game_date)

        if team_1_game_data.empty or team_2_game_data.empty:
            print(f"Skipping match due to insufficient data for teams {A_teamname} and {B_teamname}")
            continue
        
        for target in targets:
            actual_results[target].append(row[target])

        match_list_left, match_list_right = produce_match_list(team_1_game_data, team_2_game_data)
        match_list_left = match_list_left.select_dtypes(include=['number'])
        match_list_right = match_list_right.select_dtypes(include=['number'])

        X_match_tensor_left = torch.tensor(match_list_left.values, dtype=torch.float32)  # 轉換為 PyTorch 張量
        with torch.no_grad():
            model_outputs = model(X_match_tensor_left)
            if any(torch.isnan(output).any() for output in model_outputs):
                print("NaN detected in model outputs!")
                continue 
            left_wins, left_firstInhibitor, left_firstTower = model_outputs  # 解包輸出

        X_match_tensor_right = torch.tensor(match_list_right.values, dtype=torch.float32)  # 轉換為 PyTorch 張量
        with torch.no_grad():
            model_outputs = model(X_match_tensor_right)
            if any(torch.isnan(output).any() for output in model_outputs):
                print("NaN detected in model outputs!")
                continue 
            right_wins, right_firstInhibitor, right_firstTower = model_outputs  # 解包輸出
        
        # 整合機率計算
        votes = {
            "A_wins": left_wins.mean().item() + (1 - right_wins.mean().item()),
            "A_firstInhibitorKill": left_firstInhibitor.mean().item() + (1 - right_firstInhibitor.mean().item()),
            "A_firstTowerKill": left_firstTower.mean().item() + (1 - right_firstTower.mean().item()),
        }

        for key, value in votes.items():
            if np.isnan(value):
                print(f"NaN detected in votes for {key}")
                continue

        for target, vote_value in votes.items():
            predicted_results[target].append(vote_value)

    roc_auc_scores = {}
    for target in targets:
        y_true = np.array(actual_results[target])
        y_pred = np.array(predicted_results[target])
        print(len(y_true))
        print(len(y_pred))

        '''# 過濾 NaN 值
        valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]
        print(len(y_true))
        print(len(y_pred))'''

        # 計算 ROC-AUC
        if len(y_true) > 0 and len(y_pred) > 0:  # 確保數據不為空
            roc_auc_scores[target] = roc_auc_score(y_true, y_pred)
            print(f"{target} ROC-AUC: {roc_auc_scores[target]:.4f}")
        else:
            print(f"{target} 沒有足夠的有效數據計算 ROC-AUC")

    # 將 ROC-AUC 保存到 CSV
    roc_auc_df = pd.DataFrame(list(roc_auc_scores.items()), columns=["Target", "ROC-AUC"])
    roc_auc_df.to_csv("roc_auc_scores.csv", index=False)
    print("ROC-AUC 已保存至 roc_auc_scores.csv")

    return roc_auc_scores
