import requests
import pandas as pd
import os

def fetch_match_data(league_id, year, month):
    """
    Fetch match data from the API and extract specific fields.
    """
    url = "https://esports.op.gg/matches/graphql/__query__ListPagedAllMatches"
    headers = {
        "accept": "*/*",
        "accept-language": "zh-TW,zh;q=0.9,en-US;q=0.8",
        "content-type": "application/json",
        "origin": "https://esports.op.gg",
        "referer": "https://esports.op.gg/schedules",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }

    # GraphQL query parameters
    payload = {
        "operationName": "ListPagedAllMatches",
        "variables": {
            "leagueId": league_id,  # Specify league
            "year": year,           # Specify year
            "month": month,         # Specify month
            "teamId": None,
            "utcOffset": 480,
            "page": 0
        },
        "query": """
        query ListPagedAllMatches($leagueId: ID, $year: Int, $month: Int, $page: Int, $utcOffset: Int) {
          pagedAllMatches(
            leagueId: $leagueId
            year: $year
            month: $month
            page: $page
            utcOffset: $utcOffset
          ) {
            id
            homeScore
            awayScore
            homeTeam {
              name
            }
            awayTeam {
              name
            }
            scheduledAt
          }
        }
        """
    }

    # Send request and extract data
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        # Extract match data
        matches = data.get("data", {}).get("pagedAllMatches", [])
        extracted_data = []

        for match in matches:
            game_id = match.get("id")
            home_team_name = match.get("homeTeam", {}).get("name")
            away_team_name = match.get("awayTeam", {}).get("name")
            home_score = match.get("homeScore", 0)
            away_score = match.get("awayScore", 0)
            total_score = home_score + away_score
            scheduled_at = match.get("scheduledAt")

            extracted_data.append({
                "game_id": game_id,
                "home_team_name": home_team_name,
                "away_team_name": away_team_name,
                "total_score": total_score,
                "scheduled_at": scheduled_at
            })

        return extracted_data

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []

def save_to_csv(data, filename):
    """
    Save match data to a CSV file.
    """
    if not data:
        print("No match data retrieved.")
        return

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Data has been saved to {filename}")

if __name__ == "__main__":
    # User input
    league_id = input("Enter league ID (e.g., Worlds: 96, LPL: 98, LCK: 99, MSI: 100): ")
    year = int(input("Enter the year (e.g., 2024): "))

    # Fetch data for the entire year and save
    all_data = []
    for month in range(1, 13):
        month_data = fetch_match_data(league_id, year, month)
        print(f"Month {month}: Retrieved {len(month_data)} records.")  # Print the number of records retrieved for the month
        all_data.extend(month_data)

    # Save all data for the year to a CSV file
    save_to_csv(all_data, f"./GameID/matches_{league_id}_{year}_all_months.csv")

