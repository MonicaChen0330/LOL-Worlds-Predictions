# League of Legends Worlds Predictions Project

## Project Workflow
### Crawler
1. **GetGameID.py**: Fetch match IDs for the specified league and year.
2. **GetGameStats.py**: Fetch detailed game statistics for each match.
3. **MatchStatsToCSV.py**: Clean and process the raw game statistics data into CSV format.
4. **MergeCSV.py**: Combine multiple CSV files into a single dataset for model training or testing.

## Require Dependencies
Before running the scripts, install the required libraries:
```bash
pip install requests pandas numpy os json time random
```
Ensure that Python and `pip` are correctly installed and added to your system path.

## How to Run the Project
### Crawler
#### **1. Fetch Match IDs** (`GetGameID.py`)
Fetch all match IDs for a specified league and year.
##### **Usage:**
1. Run the script:
   ```bash
   python GetGameID.py
   ```
2. Input the required parameters:
   - `leagueID`: League you want to fetch (e.g., **Worlds: 96**, **LPL: 98**, **LCK: 99**, **MSI: 100**).
   - `year`: Year of the tournament (e.g., `2024`).
##### **Output:**
- A CSV file containing all match IDs is saved in:
   ```plaintext
   ./GameID/matches_{league_id}_{year}_all_months.csv
   ```

#### **2. Fetch Game Statistics** (`GetGameStats.py`)
Fetch detailed game statistics, including team names, scores, and player statistics, using the match ID CSV file generated in Step 1.
#### **Usage:**
1. Run the script:
   ```bash
   python GetGameStats.py
   ```
2. Input the required parameters:
   - `leagueID`: League you want to fetch.
   - `year`: Year of the tournament.
#### **Input:**
- Ensure the corresponding match ID CSV file exists in:
   ```plaintext
   ./GameID/matches_{league_id}_{year}_all_months.csv
   ```

#### **3. Clean Game Statistics** (`MatchStatsToCSV.py`)
Process and clean the fetched JSON files into a structured CSV format.
#### **Usage:**
1. Run the script:
   ```bash
   python MatchStatsToCSV.py
   ```
2. Input the required parameters:
   - `leagueID`: League you want to clean.
   - `year`: Year of the tournament.
3. Input the target directory name here. All data in this derectory may be combine in a single CSV file by step4.

#### **Input:**
- JSON files generated in:
   ```plaintext
   ./MatchStats/{league_name}_{year}/
   ```

#### **4. Merge CSV Files** (`MergeCSV.py`)
Combine multiple cleaned CSV files into a single dataset, suitable for training or testing machine learning models.
#### **Usage:**
1. Place all the CSV files you want to merge into the same folder.
2. Run the script:
   ```bash
   python MergeCSV.py
   ```
3. Input the required parameters:
   - Specify the **input folder** containing the CSV files.
   - Specify the output file name (e.g., `Training.csv` or `Testing.csv`).
#### **Output:**
- A combined CSV file is saved at the specified location:
   ```plaintext
   ./Training.csv or ./Testing.csv
   ```

## **Directory Structure**

```plaintext
Project Root/LOL_gamestats_crawler
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
│   ├── train/  
│   │    ├── LCK_2024.csv
│   │    ├── LPL_2024.csv
│   │    └── ...
│   ├── test/
│   └── ...
│
├── GetGameID.py            # Fetch match IDs
├── GetGameStats.py         # Fetch match statistics
├── MatchStatsToCSV.py   # Clean match data
├── MergeCSV.py             # Combine CSV files
└── README.md               # Documentation (this file)