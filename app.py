import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- Configuration ---
MODEL_FILE = 'form_model.joblib'
TEAMS_FILE = 'teams.joblib'
N_GAMES = 5 
MODEL_FEATURES = ['HT_AvgP', 'HT_AvgGS', 'HT_AvgGC', 'AT_AvgP', 'AT_AvgGS', 'AT_AvgGC']
RESULT_MAP = {2: 'Home Win (H) 🏡', 1: 'Draw (D) 🤝', 0: 'Away Win (A) ✈️'}

# --- Caching Functions (Load once) ---
@st.cache_resource
def load_model_and_data(model_path, teams_path, data_path='SP1.csv'):
    if not os.path.exists(model_path) or not os.path.exists(teams_path):
        st.error(f"Error: Required files ({model_path} or {teams_path}) not found. Run 'prep_model_form.py' first.")
        return None, None, None
    try:
        model = joblib.load(model_path)
        teams = joblib.load(teams_path)
        
        # Load the data for form calculation (must match prep script)
        df = pd.read_csv(data_path)
        df = df.rename(columns={'FTHG': 'HG', 'FTAG': 'AG', 'HomeTeam': 'HT', 'AwayTeam': 'AT'})
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        return model, teams, df
    except Exception as e:
        st.error(f"Failed to load files: {e}")
        return None, None, None

# --- Feature Engineering Functions (MUST match prep_model_form.py) ---

def get_match_results(df, team, date):
    # Matches where the team was Home or Away
    home_matches = df[(df['HT'] == team) & (df['Date'] < date)]
    away_matches = df[(df['AT'] == team) & (df['Date'] < date)]
    
    home_results = home_matches[['Date', 'HT', 'AT', 'HG', 'AG', 'FTR']].copy()
    away_results = away_matches[['Date', 'AT', 'HT', 'AG', 'HG', 'FTR']].copy()
    
    home_results.columns = ['Date', 'Team', 'Opponent', 'GS', 'GC', 'FTR']
    away_results.columns = ['Date', 'Team', 'Opponent', 'GS', 'GC', 'FTR']

    all_results = pd.concat([home_results, away_results]).sort_values('Date', ascending=False)
    return all_results

def calculate_form(df, team, n_games=N_GAMES):
    """For prediction, we use the latest date in the entire dataset to ensure we use all training data."""
    # Find the latest date in the entire dataset for calculation
    latest_date = df['Date'].max()
    past_results = get_match_results(df, team, latest_date)
    
    if past_results.empty:
        return {'AvgP': 0.0, 'AvgGS': 0.0, 'AvgGC': 0.0}

    past_results['Points'] = np.where(past_results['FTR'] == 'H', 3, 
                             np.where(past_results['FTR'] == 'D', 1, 0))

    last_n = past_results.head(n_games)
    
    return {
        'AvgP': last_n['Points'].mean(),
        'AvgGS': last_n['GS'].mean(),
        'AvgGC': last_n['GC'].mean(),
    }

# Load resources
model, unique_teams, df_historical = load_model_and_data(MODEL_FILE, TEAMS_FILE)

# --- Streamlit UI ---
st.title("⚽ Football Match Form-Based Predictor")
st.markdown("This prototype uses **Home Team** and **Away Team** names to calculate historical **Form and Strength** and predict the outcome.")

if model is not None and df_historical is not None:
    st.header("1. Select Competing Teams")

    # Sort teams for cleaner dropdown display
    sorted_teams = sorted(unique_teams)
    
    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox(
            "Select Home Team",
            sorted_teams
        )

    with col2:
        away_team = st.selectbox(
            "Select Away Team",
            [t for t in sorted_teams if t != home_team] # Filter out the home team
        )
    
    if st.button("Predict Match Outcome", type="primary"):
        # 1. Calculate features for selected teams
        h_form = calculate_form(df_historical, home_team)
        a_form = calculate_form(df_historical, away_team)
        
        # 2. Create input DataFrame
        input_data = pd.DataFrame([[
            h_form['AvgP'], h_form['AvgGS'], h_form['AvgGC'],
            a_form['AvgP'], a_form['AvgGS'], a_form['AvgGC']
        ]], columns=MODEL_FEATURES)

        # 3. Make Prediction
        prediction_int = model.predict(input_data)[0]
        prediction_text = RESULT_MAP.get(prediction_int, "Error")
        
        # Get Probabilities (Class order: 0=Away, 1=Draw, 2=Home)
        probabilities = model.predict_proba(input_data)[0]
        prob_away = probabilities[0]
        prob_draw = probabilities[1]
        prob_home = probabilities[2]
        
        # 4. Display Results
        st.subheader("✅ Predicted Outcome")
        st.success(f"**{home_team}** vs **{away_team}**: **{prediction_text}**")

        st.markdown("---")
        st.subheader("Confidence Scores")
        
        prob_col1, prob_col2, prob_col3 = st.columns(3)
        
        with prob_col1:
            st.metric(f"{home_team} Win", f"{prob_home * 100:.1f}%")
        with prob_col2:
            st.metric("Draw", f"{prob_draw * 100:.1f}%")
        with prob_col3:
            st.metric(f"{away_team} Win", f"{prob_away * 100:.1f}%")

else:
    st.warning("Please run **`prep_model_form.py`** first to generate the model and team files.")