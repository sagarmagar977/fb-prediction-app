import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

FILE_NAME = 'SP1.csv'
MODEL_FILE = 'form_model.joblib'
TEAMS_FILE = 'teams.joblib'
N_GAMES = 5 # Number of previous games to calculate form

print("Starting form-based model preparation...")

# 1. Load Data and Initial Cleanup
df = pd.read_csv(FILE_NAME)
df = df.rename(columns={'FTHG': 'HG', 'FTAG': 'AG', 'HomeTeam': 'HT', 'AwayTeam': 'AT'})
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values('Date').reset_index(drop=True)

# Map target (FTR)
result_mapping = {'H': 2, 'D': 1, 'A': 0}
df['Target'] = df['FTR'].map(result_mapping)

# Save unique teams for Streamlit dropdown
unique_teams = pd.concat([df['HT'], df['AT']]).unique().tolist()
joblib.dump(unique_teams, TEAMS_FILE)
print(f"Unique teams saved to {TEAMS_FILE}.")

# --- 2. Feature Engineering Functions (Crucial Logic) ---

def get_match_results(df, team, date):
    """Returns all past matches for a team before a specific date."""
    # Matches where the team was Home or Away
    home_matches = df[(df['HT'] == team) & (df['Date'] < date)]
    away_matches = df[(df['AT'] == team) & (df['Date'] < date)]
    
    # Standardize result data for easier calculation
    home_results = home_matches[['Date', 'HT', 'AT', 'HG', 'AG', 'FTR']].copy()
    away_results = away_matches[['Date', 'AT', 'HT', 'AG', 'HG', 'FTR']].copy()
    
    # Rename columns for consistency (Team = HT, Opponent = AT)
    home_results.columns = ['Date', 'Team', 'Opponent', 'GS', 'GC', 'FTR']
    away_results.columns = ['Date', 'Team', 'Opponent', 'GS', 'GC', 'FTR']

    all_results = pd.concat([home_results, away_results]).sort_values('Date', ascending=False)
    return all_results

def calculate_form(df, team, date, n_games=N_GAMES):
    """Calculates Avg Points, Avg Goals Scored/Conceded for the last N games."""
    past_results = get_match_results(df, team, date)
    
    if past_results.empty:
        # Default features for the very first games
        return {'AvgP': 0.0, 'AvgGS': 0.0, 'AvgGC': 0.0}

    # Points Calculation
    past_results['Points'] = np.where(past_results['FTR'] == 'H', 3, 
                             np.where(past_results['FTR'] == 'D', 1, 0))

    # Calculate metrics over the last N games
    last_n = past_results.head(n_games)
    
    return {
        'AvgP': last_n['Points'].mean(),
        'AvgGS': last_n['GS'].mean(),
        'AvgGC': last_n['GC'].mean(),
    }

# --- 3. Apply Feature Engineering to DataFrame ---

features_list = []

# Iteratively calculate features for every match using historical data
for index, row in df.iterrows():
    home_form = calculate_form(df, row['HT'], row['Date'], N_GAMES)
    away_form = calculate_form(df, row['AT'], row['Date'], N_GAMES)
    
    features = {
        'HT_AvgP': home_form['AvgP'],
        'HT_AvgGS': home_form['AvgGS'],
        'HT_AvgGC': home_form['AvgGC'],
        'AT_AvgP': away_form['AvgP'],
        'AT_AvgGS': away_form['AvgGS'],
        'AT_AvgGC': away_form['AvgGC']
    }
    features_list.append(features)

df_features = pd.DataFrame(features_list)
df = pd.concat([df, df_features], axis=1).dropna(subset=['Target'])

# Define final feature set
MODEL_FEATURES = ['HT_AvgP', 'HT_AvgGS', 'HT_AvgGC', 'AT_AvgP', 'AT_AvgGS', 'AT_AvgGC']

# Drop rows that don't have enough history to calculate form (if N_GAMES is large)
# However, the calculate_form handles the start of the season by returning 0.0
X = df[MODEL_FEATURES]
y = df['Target']

# --- 4. Model Training and Saving ---
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier on Form/Strength features...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Training complete.")

joblib.dump(model, MODEL_FILE)
print(f"Model successfully trained and saved as {MODEL_FILE}")