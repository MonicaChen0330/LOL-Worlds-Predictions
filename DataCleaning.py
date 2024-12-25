import pandas as pd

player_list = ["top", "mid", "adc", "sup", "jun"]
team = ["red", "blue"]
role_task = ["wardsPlaced", "wardsKilled", "sightWardsBought", "goldEarned", 
        "totalDamageDealt", "firstBloodAssist", "firstInhibitorAssist",
        "firstTowerAssist"]
team_task = ["kills", "deaths", "assists", "tower_kills", "inhibitor_kills", "gold_earned"]
basic = ["game_id", "game_length"]

lol_train_data = pd.read_csv("./LOL_gamestats_crawler/train.csv")
lol_test_data = pd.read_csv("./LOL_gamestats_crawler/worlds_test.csv")


'''
step 0
把basic list中的基本屬性drop掉
'''
for b in basic:
    lol_train_data.drop(columns=[b], inplace=True)
    lol_test_data.drop(columns=[b], inplace=True)
#lol_train_data.drop(columns=["game_date"], inplace=True)

'''
step 1
ReCalculate Dragon Kills
'''
lol_train_data['blue_dragon_kills'] -= lol_train_data['blue_elder_dragon_kills']
lol_train_data['red_dragon_kills'] -= lol_train_data['red_elder_dragon_kills']
lol_test_data['blue_dragon_kills'] -= lol_test_data['blue_elder_dragon_kills']
lol_test_data['red_dragon_kills'] -= lol_test_data['red_elder_dragon_kills']

'''
step 2
把各隊各路 visionWardsBought 改成 sight
'''
for color in team:
    for road in player_list:
        column_name = f"{color}_{road}_visionWardsBought"
        new_column_name = f"{color}_{road}_sight"
        if column_name in lol_train_data.columns:
            lol_train_data.rename(columns={column_name: new_column_name}, inplace=True)
        if column_name in lol_test_data.columns:
            lol_test_data.rename(columns={column_name: new_column_name}, inplace=True)   

'''
step 3
把team_task中團隊共同任務刪除
'''
for color in team:
    for t in team_task:
        column_name = f"{color}_{t}"
        if column_name in lol_train_data.columns:
            lol_train_data.drop(columns=[column_name], inplace=True)
        if column_name in lol_test_data.columns:
            lol_test_data.drop(columns=[column_name], inplace=True)

'''
step 4
把紅藍方各路共同任務刪除
'''
for color in team:
    for road in player_list:
        for t in role_task:
            column_name = f"{color}_{road}_{t}"
            if column_name in lol_train_data.columns:
                lol_train_data.drop(columns=[column_name], inplace=True)
            if column_name in lol_test_data.columns:
                lol_test_data.drop(columns=[column_name], inplace=True)

'''
step 5
挑選最重要的首塔作為預測目標，並刪除關於防禦塔的相似特徵
firstTowerKill-> A_firstTowerKill(bool)
a. 統計某方首塔
b. 創建新特徵 "A_firstTowerKill"
c. 12/17 補上新特徵 "B_firstTowerKill"。兩隊拆開來時才有各自是否該局得到首塔首兵營的數據
d. 12/25 刪除相似度過高的預測目標_firstInhibitorKill
'''
columns_to_check = [f"blue_{road}_firstTowerKill" for road in player_list]
lol_train_data["A_firstTowerKill"] = (lol_train_data[columns_to_check].sum(axis=1) == 1).astype(int)
lol_test_data["A_firstTowerKill"] = (lol_test_data[columns_to_check].sum(axis=1) == 1).astype(int)

lol_train_data["B_firstTowerKill"] = (lol_train_data["A_firstTowerKill"] == 0).astype(int)
lol_test_data["B_firstTowerKill"] = (lol_test_data["A_firstTowerKill"] == 0).astype(int)

first_task = ["firstInhibitorKill", "firstTowerKill", "firstInhibitorAssist", "firstTowerAssist"]
for color in team:
    for road in player_list:
        for t in first_task:
            column_name = f"{color}_{road}_{t}"
            if column_name in lol_train_data.columns:
                lol_train_data.drop(columns=[column_name], inplace=True)
            if column_name in lol_test_data.columns:
                lol_test_data.drop(columns=[column_name], inplace=True)

"""
step 6
不考慮場地紅藍方ban pick機制，因此將feature名稱中的red/bule替換成A/B
"""
# Change blue_ to A_ and red_ to B_
lol_train_data.columns = [col.replace('blue_', 'A_') if col.startswith('blue_') else col for col in lol_train_data.columns]
lol_train_data.columns = [col.replace('red_', 'B_') if col.startswith('red_') else col for col in lol_train_data.columns]
lol_test_data.columns = [col.replace('blue_', 'A_') if col.startswith('blue_') else col for col in lol_test_data.columns]
lol_test_data.columns = [col.replace('red_', 'B_') if col.startswith('red_') else col for col in lol_test_data.columns]

"""
step 7
每局比賽中只會有一位玩家獲得首殺，因此將這A/B_Role_firstBloodKill十個特徵合併成A_firstBlood，並將其作為其中一個預測目標
"""
columns_A_firstBlood = [f"A_{role}_firstBloodKill" for role in player_list]
columns_B_firstBlood = [f"B_{role}_firstBloodKill" for role in player_list]

lol_train_data["A_firstBlood"] = (lol_train_data[columns_A_firstBlood].sum(axis=1) == 1).astype(int)
lol_train_data["B_firstBlood"] = (lol_train_data["A_firstBlood"] == 0).astype(int)
lol_train_data.drop(columns=columns_A_firstBlood + columns_B_firstBlood, inplace=True)
lol_test_data["A_firstBlood"] = (lol_test_data[columns_A_firstBlood].sum(axis=1) == 1).astype(int)
lol_test_data["B_firstBlood"] = (lol_test_data["A_firstBlood"] == 0).astype(int)
lol_test_data.drop(columns=columns_A_firstBlood + columns_B_firstBlood, inplace=True)

"""
step 8
刪除只有特定賽區有的特徵: _opScore
刪除因改版而新增的地圖物件特徵: _voidgrubs
"""
op_scores = [col for col in lol_train_data.columns if col.endswith('opScore')]
lol_train_data.drop(op_scores + ['A_voidgrubs', 'B_voidgrubs'], axis=1, inplace=True)
lol_test_data.drop(op_scores + ['A_voidgrubs', 'B_voidgrubs'], axis=1, inplace=True)

lol_train_data.to_csv("train_lol_cleaned.csv", index=False)
lol_test_data.to_csv("test_lol_cleaned.csv", index=False)
