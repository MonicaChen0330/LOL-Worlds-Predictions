import os
import json
import pandas as pd

# Prompt the user to input league and year
league = input("Enter League ID (Worlds, LPL, LCK, MSI):").strip()
year = input("Enter the year (e.g. 2024): ").strip()
output_dir = input(f"Enter your output directory name here (e.g.Train, All data in this file may be combine in a single CSV file later):") #Set the directory name here
output_dir = './StatsCSV/' + output_dir
os.makedirs(output_dir, exist_ok=True) #Ensure the output file exist

# Define input and output paths
input_folder = f'./MatchStats/{league}_{year}'
output_file = f'{output_dir}/{league}_{year}.csv'


# Initialize an empty list to store processed data
all_data = []

# Loop through all JSON files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.json'):  # Only process JSON files
        file_path = os.path.join(input_folder, filename)
        
        # Load JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Extract key data
        game_data = data['data']['gameByMatch']
        if not game_data:  # Skip the file if gameByMatch is null
                print(f"Skipping file {filename}: 'gameByMatch' data is null.")
                continue
        teams = game_data['teams']
        players = game_data['players']
        object_kills = game_data['objectKills']
        
        # Determine the winner team
        winner_team_id = game_data['winner']['id']
        blue_wins = 1 if teams[0]['team']['id'] == winner_team_id else 0
        
        # Initialize data structures
        horde_counts = {'blue': 0, 'red': 0}
        
        # Process object kills for HORDE counts
        for obj in object_kills:
            if obj['objectType'] == 'HORDE':
                if obj['teamId'] == teams[0]['team']['id']:  # Blue team ID
                    horde_counts['blue'] += 1
                elif obj['teamId'] == teams[1]['team']['id']:  # Red team ID
                    horde_counts['red'] += 1
        
        # Process team and player data
        merged_data = {
            'game_id': game_data['id'],  # Insert game_id
            'game_date': game_data['beginAt'],  #Insert game_date
            'game_length': game_data['length']  # Insert game_length
        }
        
        position_order = ['top', 'jun', 'mid', 'adc', 'sup']
        for i, team in enumerate(teams):
            side = team['side']
            
            # Collect all player stats for the team
            player_stats = [
                {
                    f"{side}_{player['position']}_{stat}": (
                        1 if isinstance(player[stat], bool) and player[stat] else 0  # Convert boolean to binary
                        if isinstance(player[stat], bool) else player[stat]  # Keep other values as is
                    )
                    for stat in [
                        'kills', 'deaths', 'assists', 'opScore', 'level', 'minionsKilled', 'wardsPlaced',
                        'wardsKilled', 'sightWardsBought', 'visionWardsBought', 'turretsKilled', 
                        'goldEarned', 'goldSpent', 'totalHeal', 'largestMultiKill', 'largestKillingSpree',
                        'totalDamageDealt', 'totalDamageDealtToChampions', 'totalDamageTaken',
                        'totalTimeCrowdControlDealt', 'firstBloodAssist', 'firstBloodKill', 
                        'firstInhibitorAssist', 'firstInhibitorKill', 'firstTowerAssist', 'firstTowerKill'
                    ]
                }
                for player in sorted(players, key=lambda x: position_order.index(x['position']) if x['position'] in position_order else len(position_order))
                if player['side'] == side
            ]
            
            # Flatten player stats into a single dictionary
            player_data_flat = {}
            for player_dict in player_stats:
                player_data_flat.update(player_dict)
            
            # Collect team stats
            team_stats = {
                f"{side}_teamname": team['team']['acronym'],
                f"{side}_kills": team['kills'],
                f"{side}_deaths": team['deaths'],
                f"{side}_assists": team['assists'],
                f"{side}_tower_kills": team['towerKills'],
                f"{side}_inhibitor_kills": team['inhibitorKills'],
                f"{side}_herald_kills": team['heraldKills'],
                f"{side}_dragon_kills": team['dragonKills'],
                f"{side}_elder_dragon_kills": team['elderDrakeKills'],
                f"{side}_baron_kills": team['baronKills'],
                f"{side}_gold_earned": team['goldEarned'],
                f"{side}_voidgrubs": horde_counts[side]  # Rename HORDE Counts to voidgrubs
            }
            
            # Combine team and player stats
            team_stats.update(player_data_flat)
            merged_data.update(team_stats)
        
        # Add blue_wins column
        merged_data['blue_wins'] = blue_wins
        
        # Append to the list
        all_data.append(merged_data)

# Convert the collected data into a DataFrame
df = pd.DataFrame(all_data)

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)

print(f"Processing complete. Consolidated data saved to {output_file}.")
