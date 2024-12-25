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
firstInhibitorKill-> A_firstInhibitorKill(bool)
firstTowerKill-> A_firstInhibitorKill(bool)
a. 統計某方首塔及首兵營
b. 創建新特徵 "A_firstTowerKill", "A_firstInhibitorKill"
c. 12/17 補上新特徵 "B_firstTowerKill", "B_firstInhibitorKill"。兩隊拆開來時才有各自是否該局得到首塔首兵營的數據
'''
columns_to_check = [f"blue_{road}_firstInhibitorKill" for road in player_list]
lol_train_data["A_firstInhibitorKill"] = (lol_train_data[columns_to_check].sum(axis=1) == 1).astype(int)
lol_test_data["A_firstInhibitorKill"] = (lol_test_data[columns_to_check].sum(axis=1) == 1).astype(int)

lol_train_data["B_firstInhibitorKill"] = (lol_train_data["A_firstInhibitorKill"] == 0).astype(int)
lol_test_data["B_firstInhibitorKill"] = (lol_test_data["A_firstInhibitorKill"] == 0).astype(int)

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

# 6
# Change blue_ to A_ and red_ to B_
lol_train_data.columns = [col.replace('blue_', 'A_') if col.startswith('blue_') else col for col in lol_train_data.columns]
lol_train_data.columns = [col.replace('red_', 'B_') if col.startswith('red_') else col for col in lol_train_data.columns]
lol_test_data.columns = [col.replace('blue_', 'A_') if col.startswith('blue_') else col for col in lol_test_data.columns]
lol_test_data.columns = [col.replace('red_', 'B_') if col.startswith('red_') else col for col in lol_test_data.columns]

# 7
# Consolidate First Blood Kill
columns_A_firstBlood = [f"A_{role}_firstBloodKill" for role in player_list]
columns_B_firstBlood = [f"B_{role}_firstBloodKill" for role in player_list]

lol_train_data["A_firstBlood"] = (lol_train_data[columns_A_firstBlood].sum(axis=1) == 1).astype(int)
lol_train_data["B_firstBlood"] = (lol_train_data["A_firstBlood"] == 0).astype(int)
lol_train_data.drop(columns=columns_A_firstBlood + columns_B_firstBlood, inplace=True)
lol_test_data["A_firstBlood"] = (lol_test_data[columns_A_firstBlood].sum(axis=1) == 1).astype(int)
lol_test_data["B_firstBlood"] = (lol_test_data["A_firstBlood"] == 0).astype(int)
lol_test_data.drop(columns=columns_A_firstBlood + columns_B_firstBlood, inplace=True)

# 8
# Drop relevancy less than 0.002
least_relevancy_feature = ['top_totalHeal', 'sup_sight', 'sup_totalHeal', 'adc_sight', 'mid_sight',
                            'mid_largestMultiKill', 'top_sight', 'jun_totalHeal', 'adc_totalHeal',
                            'jun_largestKillingSpree', 'top_largestMultiKill', 'mid_totalHeal',
                            'elder_dragon_kills', 'sup_kills', 'jun_largestMultiKill', 'herald_kills',
                            'sup_largestKillingSpree', 'sup_largestMultiKill']
for side in ["A", "B"]:
    for feat in least_relevancy_feature:
        column_name = f"{side}_{feat}"
        if column_name in lol_train_data.columns:
            lol_train_data.drop(columns=[column_name], inplace=True)
        if column_name in lol_test_data.columns:
            lol_test_data.drop(columns=[column_name], inplace=True)

lol_train_data.to_csv("train_lol_cleaned.csv", index=False)
lol_test_data.to_csv("test_lol_cleaned.csv", index=False)

