# League of Legends Worlds Predictions Project

## Project Workflow
1. **GetGameID.py**: Fetch match IDs for the specified league and year.
2. **GetGameStats.py**: Fetch detailed game statistics for each match.
3. **MatchStatsCleaning.py**: Clean and process the raw game statistics data into CSV format.
4. **mergeCSV.py**: Combine multiple CSV files into a single dataset for model training or testing.

## Require Dependencies
Before running the scripts, install the required libraries:
- requests
- os
- pandas
- json
- time
- random

## How to Run the Project
### Crawl
**1. Fetch Match IDs** (`GetGameID.py`): Fetch all match IDs for a specified league and year.
- Run the script:
   ```bash
   python GetGameID.py
   ```
- Input: The specify *leagueID* and *year* that you want to fetch.
- Output: A CSV file containing all match IDs is saved in
   ```plaintext
   ./GameID/matches_{league_id}_{year}_all_months.csv
   ```

2. **2. Fetch Game Statistics** (`GetGameStats.py`): Fetch detailed game statistics, including team names, scores, and player statistics, using the match ID CSV file generated in Step 1.
- Run the script:
```bash
   python GetGameStats.py
   ```
- Input: The specify leagueID and year that you want to fetch. Ensure the corresponding gameID csv file exists.
- Output: JSON files for each game are saved in the output directory
   ```plaintext
   ./MatchStats/{league_name}_{year}/
   ```

**3. Clean Game Statistics** (`MatchStatsCleaning.py`): Process and clean the fetched JSON files into a structured CSV format.
- Run the script:
   ```bash
   python MatchStatsCleaning.py
   ```
- Input: The specify leagueID and year that you want to clean.
- Output: A clean CSV file with processed match statistics:
   ```plaintext
   ./StatsCSV/{league_name}_{year}.csv
   ```

**4. Merge CSV Files** (`mergeCSV.py`): Combine multiple cleaned CSV files into a single dataset, suitable for training or testing machine learning models.
- Place all the CSV files you want to merge into the same folder.
- Run the script:
   ```bash
   python mergeCSV.py
   ```
- Input the required parameters:
   - Specify the **input folder** containing the CSV files.
   - Specify the output file name (e.g., `Training.csv` or `Testing.csv`).
- Output: A combined CSV file is saved at the specified location
   ```plaintext
   ./Training.csv or ./Testing.csv

## **Directory Structure**

```plaintext
Project Root
│
├── GameID/                 # Match ID CSV files
│   ├── matches_99_2024_all_months.csv
│   └── ...
│
├── MatchStats/             # Raw JSON files of match statistics
│   ├── LCK_2024/
│   ├── LPL_2024/
│   └── ...
│
├── StatsCSV/               # Cleaned and structured CSV files
│   ├── LCK_2024.csv
│   ├── LPL_2024.csv
│   └── ...
│
├── GetGameID.py            # Fetch match IDs
├── GetGameStats.py         # Fetch match statistics
├── MatchStatsCleaning.py   # Clean match data
├── mergeCSV.py             # Combine CSV files
└── README.md               # Documentation (this file)