# League of Legends Worlds Predictions Project

## Overview👁️
This project aims to predict the outcomes of League of Legends (LoL) World Championship matches using machine learning models. The project consists of two main components: a web crawler to fetch match data and a machine learning model to predict match outcomes.

## Installation🧰
To run the project, you need to install the required dependencies. You can install them using the following command:
```bash
pip install -r requirements.txt
```

## Project Workflow🔄
The project consists of two main components: a web crawler to fetch match data and a machine learning model to predict match outcomes. The workflow is as follows:
1. **Crawler Component**:
   - **GetGameID.py**: Fetch match IDs for a specified league and year.
   - **GetGameStats.py**: Fetch detailed game statistics for each match.
   - **MatchStatsToCSV.py**: Clean and process the raw game statistics data into CSV format.
   - **MergeCSV.py**: Combine multiple CSV files into a single dataset for model training or testing.
2. **Machine Learning Model**:
   - **DataCleaning.py**: Clean and preprocess the raw dataset for training the machine learning model.
   - **FeatureAnalysis.ipynb**: Analyze the features in the dataset to identify important predictors for the model.
   - **lol.py**: Train a machine learning model using the cleaned dataset.
   - **prediction.py**: Define the prediction function to predict match outcomes for future matches.
   - **model.py**: Provide three kinds of models for training and prediction.

## Usage🚀
### Crawler🐞
The crawler component fetches match data from the Riot Games API. To fetch match data, follow these steps:
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

### Machine Learning Model🤖
The machine learning component trains a model to predict match outcomes using the cleaned dataset. The workflow is as follows:

#### 1. Data Cleaning (`DataCleaning.py`)
Clean and preprocess the raw dataset for training the machine learning model.
#### **Usage:**
1. Run the script:
   ```bash
   python DataCleaning.py
   ```
2. The script will clean the dataset and save the cleaned data in two CSV files:
   - `train_lol.csv`: Training dataset.
   - `test_lol.csv`: Testing dataset.

#### 2. Feature Analysis (`FeatureAnalysis.ipynb`)
Analyze the features in the dataset to identify important predictors for the model.
#### **Usage:**
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook FeatureAnalysis.ipynb
   ```
2. Run the cells in the notebook to analyze the features.
3. Visualize the feature importance and correlation with the target variable.
4. Identify the important features for the model.
5. The notebook will generate the dataset with selected features:
   - `train_lol_selected.csv`: Training dataset with selected features.
   - `test_lol_selected.csv`: Testing dataset with selected features.

#### 3. Model Training (`lol.py`, `model.py`)
Train a machine learning model using the cleaned dataset.
#### **Usage:**
1. Run the script:
   ```bash
   python lol.py
   ```
2. The script will train the model and test it on the testing dataset.

#### **Notice:**
- You can change the model by referring to the `model.py` file.

#### 4. Model Prediction (`lol.py`, `prediction.py`)
Use the trained model to predict match outcomes for future matches.
#### **Usage:**
1. Run the script:
   ```bash
   python lol.py
   ```
2. The script will using the prediction function to predict the outcome of the matches.
3. The prediction results will be displayed in the console, saved in a CSV file, and plotted in a graph.

## Project Structure📂

```plaintext
Project Root/
│
├── LOL_gamestats_crawler
│   │
│   ├── GameID/                 # Match ID CSV files
│   │   ├── matches_99_2024_all_months.csv
│   │   └── ...
│   │
│   ├── MatchStats/             # Raw JSON files of match statistics
│   │   ├── LCK_2024/
│   │   ├── LPL_2024/
│   │   └── ...
│   │
│   ├── StatsCSV/               # Cleaned and structured CSV files
│   │   ├── train/  
│   │   │    ├── LCK_2024.csv
│   │   │    ├── LPL_2024.csv
│   │   │    └── ...
│   │   ├── test/
│   │   └── ...
│   │
│   ├── GetGameID.py            # Fetch match IDs
│   ├── GetGameStats.py         # Fetch match statistics
│   ├── MatchStatsToCSV.py   # Clean match data
│   ├── MergeCSV.py             # Combine CSV files
│   └── README.md               # Documentation (this file)
│
├── DataCleaning.py
├── FeatureAnalysis.ipynb
├── lol.py
├── model.py
├── ModelScorePlot.py
├── prediction.py
├── README.md
├── requirements.txt
├── train_lol_selected.csv
└── test_lol_selected.csv
```
