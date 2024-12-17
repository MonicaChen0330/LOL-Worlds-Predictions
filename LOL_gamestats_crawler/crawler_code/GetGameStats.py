import pandas as pd
import os
import time
import random
import requests
import json

# League ID to League Name Mapping
LEAGUE_NAME_MAP = {
    '96': 'Worlds',
    '98': 'LPL',
    '99': 'LCK',
    '100': 'MSI',
}

def fetch_game_data(match_id, set_number, output_dir):
    """
    Fetch game data from API and save as JSON.
    Avoid sending requests if the file already exists.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the filename dynamically
    filename = os.path.join(output_dir, f"{match_id}_set{set_number}.json")

    # Skip if the file already exists
    if os.path.exists(filename):
        print(f"File already exists: {filename}, skipping...")
        return

    # API URL and headers
    url = "https://esports.op.gg/matches/graphql/__query__GetGameByMatch"
    headers = {
        "accept": "*/*",
        "accept-language": "zh-TW,zh;q=0.9,ja-JP;q=0.8,ja;q=0.7,en-US;q=0.6,en;q=0.5,zh-CN;q=0.4",
        "apollo-require-preflight": "true",
        "content-type": "application/json",
        "cookie": "your-valid-cookie-here",  # Replace with a valid cookie if needed
        "origin": "https://esports.op.gg",
        "referer": f"https://esports.op.gg/matches/{match_id}",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }

    # GraphQL query payload
    payload = {
        "operationName": "GetGameByMatch",
        "variables": {"matchId": str(match_id), "set": set_number},
        "query": """
        query GetGameByMatch($matchId: ID!, $set: Int) {
            gameByMatch(matchId: $matchId, set: $set) {
                id
                beginAt
                length
                winner {
                    id
                }
                teams {
                    team {
                        id
                        acronym
                    }
                    side
                    kills
                    deaths
                    assists
                    towerKills
                    inhibitorKills
                    heraldKills
                    dragonKills
                    elderDrakeKills
                    baronKills
                    goldEarned
                }
                objectKills {
                    teamId
                    objectType
                }
                players {
                    side
                    position
                    kills
                    deaths
                    assists
                    opScore
                    level
                    minionsKilled
                    wardsPlaced
                    wardsKilled
                    sightWardsBought
                    visionWardsBought
                    turretsKilled
                    goldEarned
                    goldSpent
                    totalHeal
                    largestMultiKill
                    largestKillingSpree
                    totalDamageDealt
                    totalDamageDealtToChampions
                    totalDamageTaken
                    totalTimeCrowdControlDealt
                    firstBloodAssist
                    firstBloodKill
                    firstInhibitorAssist
                    firstInhibitorKill
                    firstTowerAssist
                    firstTowerKill
                }
            }
        }
        """
    }

    try:
        # Send the request
        response = requests.post(url, headers=headers, json=payload)

        # Handle HTTP 429 Too Many Requests
        if response.status_code == 429:
            print("Too many requests. Waiting for 5 seconds before retrying...")
            time.sleep(5)
            return fetch_game_data(match_id, set_number, output_dir)

        # Raise exception for other HTTP errors
        response.raise_for_status()

        # Parse and save the response
        data = response.json()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data successfully saved: {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data for match_id={match_id}, set_number={set_number}. Error: {e}")


if __name__ == "__main__":
    # User inputs
    league_id = input("Enter League ID (Worlds(96), LPL(98), LCK(99), MSI(100)): ").strip()
    year = input("Enter Year (e.g., 2024): ").strip()

    # Validate league_id
    league_name = LEAGUE_NAME_MAP.get(league_id)
    if not league_name:
        print(f"Invalid League ID: {league_id}. Exiting.")
        exit()

    # Set output directory dynamically
    output_dir = f"./MatchStats/{league_name}_{year}"

    # Input CSV file path
    input_file = f"./GameID/matches_{league_id}_{year}_all_months.csv"

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        exit()

    # Read the CSV file
    data = pd.read_csv(input_file)

    # Iterate through the rows and fetch game data
    for index, row in data.iterrows():
        game_id = row['game_id']
        total_score = row['total_score']

        for set_number in range(1, total_score + 1):
            fetch_game_data(game_id, set_number, output_dir)

            # Random delay between requests to avoid rate limits
            delay = random.uniform(1, 3)
            print(f"Sleeping for {delay:.2f} seconds to avoid rate limits...")
            time.sleep(delay)

    print("All data has been processed.")
