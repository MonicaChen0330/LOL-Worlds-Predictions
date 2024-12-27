import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

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

def find_most_recent_games(df, team_name, n_games, compete_date="2024-10-18T00:00:00.000Z"):
    """
    Find the most recent games before compete date for a given team
    """
    if compete_date is not None:
        df = df[df["game_date"] <= compete_date]
    team_games = df[(df["teamname"] == team_name)]
    return team_games.head(n_games)

def produce_match_list(team_data_1, team_data_2):
    """
    Based on the data of two teams, combine each other to produce new match list
    """
    match_list = []
    for _, game_1 in team_data_1.iterrows():
        for _, game_2 in team_data_2.iterrows():
            # set game_1 columns to all have "a_" prefix, and game_2 columns to all have "b_" prefix
            game_1 = game_1.rename(lambda x: "a_" + x if not x.startswith("a_") else x)
            game_2 = game_2.rename(lambda x: "b_" + x if not x.startswith("b_") else x)
            # combine them into a single row
            match = pd.concat([game_1, game_2])
            match_list.append(match)

    match_list = pd.DataFrame(match_list)
    match_list.drop(columns=["a_teamname", "b_teamname", "a_game_date", "b_game_date", "a_firstBlood", "b_firstBlood", "a_firstTowerKill", "b_firstTowerKill"], inplace=True)
    return match_list

def predictor(test_data, model, model_name):
    #print(f"test data: {test_data.shape}")
    team_data = convert_to_teamdata(test_data)
    #print(f"team data: {team_data.shape}")
    team_data = team_data.loc[:, ~team_data.columns.duplicated()]

    # 定義目標
    targets = ["A_wins", "A_firstBlood", "A_firstTowerKill"]
    actual_results = {target: [] for target in targets}
    predicted_results_roc = {target: [] for target in targets}
    predicted_results_acc = {target: [] for target in targets}

    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing CSV file"):
        A_teamname = row["A_teamname"]
        B_teamname = row["B_teamname"]

        team_1_game_data = find_most_recent_games(team_data, A_teamname, 3)
        team_2_game_data = find_most_recent_games(team_data, B_teamname, 3)

        if team_1_game_data.empty or team_2_game_data.empty:
            print(f"Skipping match due to insufficient data for teams {A_teamname} and {B_teamname}")
            continue
        
        for target in targets:
            actual_results[target].append(row[target])

        match_list_left = produce_match_list(team_1_game_data, team_2_game_data)
        match_list_right = produce_match_list(team_2_game_data, team_1_game_data)

        X_match_tensor_left = torch.tensor(match_list_left.values, dtype=torch.float32)  # 轉換為 PyTorch 張量
        with torch.no_grad():
            model_outputs = model(X_match_tensor_left)
            left_wins, left_firstBlood, left_firstTower = model_outputs  # 解包輸出

        X_match_tensor_right = torch.tensor(match_list_right.values, dtype=torch.float32)  # 轉換為 PyTorch 張量
        with torch.no_grad():
            model_outputs = model(X_match_tensor_right)
            right_wins, right_firstBlood, right_firstTower = model_outputs  # 解包輸出
        
        # 整合機率計算
        votes = {
            "A_wins": (left_wins.mean().item() + (1 - right_wins.mean().item())) / 2,
            "A_firstBlood": (left_firstBlood.mean().item() + (1 - right_firstBlood.mean().item())) / 2,
            "A_firstTowerKill": (left_firstTower.mean().item() + (1 - right_firstTower.mean().item())) / 2,
        }

        for target, vote_value in votes.items():
            predicted_results_roc[target].append(vote_value)
            predicted_results_acc[target].append(1 if vote_value > 0.5 else 0)  # 轉換為二元預測結果

    roc_auc_scores = {}
    accuracy_scores = {}
    for target in targets:
        y_true = np.array(actual_results[target])
        y_pred_roc = np.array(predicted_results_roc[target])
        y_pred_acc = np.array(predicted_results_acc[target])

        # 計算 ROC-AUC 和 Accuracy
        if len(y_true) > 0 and len(y_pred_roc) > 0 and len(y_pred_acc) > 0:  # 確保數據不為空
            roc_auc_scores[target] = roc_auc_score(y_true, y_pred_roc)
            accuracy_scores[target] = accuracy_score(y_true, y_pred_acc)
        else:
            roc_auc_scores[target] = None
            accuracy_scores[target] = None

    # 保存結果至 CSV
    results_df = pd.DataFrame([{
        "Model": model_name,
        "A_wins_acc": accuracy_scores.get("A_wins", None),
        "A_firstBlood_acc": accuracy_scores.get("A_firstBlood", None),
        "A_firstTowerKill_acc": accuracy_scores.get("A_firstTowerKill", None),
        "A_wins_roc": roc_auc_scores.get("A_wins", None),
        "A_firstBlood_roc": roc_auc_scores.get("A_firstBlood", None),
        "A_firstTowerKill_roc": roc_auc_scores.get("A_firstTowerKill", None)
    }])
    file_path = "results.csv"
    if os.path.exists(file_path):
        results_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(file_path, mode='w', header=True, index=False)
    print("Results saved to results.csv")

    return roc_auc_scores, accuracy_scores

def quarterfinal_predictor(test_data, model, quarterfinal_matches, history_match, compete_date):
    team_data = convert_to_teamdata(test_data)
    team_data = team_data.loc[:, ~team_data.columns.duplicated()]

    # 定義目標
    targets = ["A_wins", "A_firstBlood", "A_firstTowerKill"]
    actual_results = {target: [] for target in targets}
    predicted_results_roc = {target: [] for target in targets}
    predicted_results_acc = {target: [] for target in targets}

    for A_teamname, B_teamname in quarterfinal_matches:
        print(f"Processing match between {A_teamname} and {B_teamname}")
        compete_date = pd.to_datetime(compete_date)

        team_1_game_data = find_most_recent_games(team_data, A_teamname, history_match, compete_date=compete_date)
        team_2_game_data = find_most_recent_games(team_data, B_teamname, history_match, compete_date=compete_date)

        if team_1_game_data.empty or team_2_game_data.empty:
            print(f"Skipping match due to insufficient data for teams {A_teamname} and {B_teamname}")
            continue

        match_list_left = produce_match_list(team_1_game_data, team_2_game_data)
        match_list_right = produce_match_list(team_2_game_data, team_1_game_data)

        X_match_tensor_left = torch.tensor(match_list_left.values, dtype=torch.float32)  # 轉換為 PyTorch 張量
        with torch.no_grad():
            model_outputs = model(X_match_tensor_left)
            if any(torch.isnan(output).any() for output in model_outputs):
                print("NaN detected in model outputs!")
                continue 
            left_wins, left_firstBlood, left_firstTower = model_outputs  # 解包輸出

        X_match_tensor_right = torch.tensor(match_list_right.values, dtype=torch.float32)  # 轉換為 PyTorch 張量
        with torch.no_grad():
            model_outputs = model(X_match_tensor_right)
            if any(torch.isnan(output).any() for output in model_outputs):
                print("NaN detected in model outputs!")
                continue 
            right_wins, right_firstBlood, right_firstTower = model_outputs  # 解包輸出

        # 整合機率計算
        votes = {
            "A_wins": (left_wins.mean().item() + (1 - right_wins.mean().item())) / 2,
            "A_firstBlood": (left_firstBlood.mean().item() + (1 - right_firstBlood.mean().item())) / 2,
            "A_firstTowerKill": (left_firstTower.mean().item() + (1 - right_firstTower.mean().item())) / 2,
        }

        for target, vote_value in votes.items():
            predicted_results_roc[target].append(vote_value)
            if target == "A_wins":
                predicted_results_acc[target].append(1 if vote_value > 0.5 else 0)
            else:
                predicted_results_acc[target].append(vote_value)
        
        print(f"Predicted probabilities: {predicted_results_acc}")
    match_winner = []
    first_tower = {}
    first_blood = {}
    for i in range(len(predicted_results_acc["A_wins"])):
        if predicted_results_acc["A_wins"][i] == 1:
            match_winner.append(quarterfinal_matches[i][0])
        else:
            match_winner.append(quarterfinal_matches[i][1])
        first_blood[quarterfinal_matches[i][0]] = predicted_results_acc["A_firstBlood"][i]
        first_blood[quarterfinal_matches[i][1]] = 1 - predicted_results_acc["A_firstBlood"][i]
        first_tower[quarterfinal_matches[i][0]] = predicted_results_acc["A_firstTowerKill"][i]
        first_tower[quarterfinal_matches[i][1]] = 1 - predicted_results_acc["A_firstTowerKill"][i]
    return match_winner, first_blood, first_tower