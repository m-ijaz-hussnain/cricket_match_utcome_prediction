import pandas as pd

# Load datasets
dim_match_summary = pd.read_excel("Cricket_Dashboard/Cricket_Data/dim_match_summary.xlsx")
dim_players = pd.read_excel("Cricket_Dashboard/Cricket_Data/dim_players.xlsx")
fact_bating_summary = pd.read_excel("Cricket_Dashboard/Cricket_Data/fact_bating_summary.xlsx")
fact_bowling_summary = pd.read_excel("Cricket_Dashboard/Cricket_Data/fact_bowling_summary.xlsx")

def load_data():
    print("dim_match_summary shape:", dim_match_summary.shape)
    print("dim_players shape:", dim_players.shape)
    print("fact_bating_summary shape:", fact_bating_summary.shape)
    print("fact_bowling_summary shape:", fact_bowling_summary.shape)

def preprocess_match_outcome_data():
    # Merge batting and bowling summaries to get team performance per match
    batting_summary = fact_bating_summary.groupby(['match', 'teamInnings']).agg({
        'runs': 'sum',
        'balls': 'sum',
        '4s': 'sum',
        '6s': 'sum'
    }).reset_index()

    bowling_summary = fact_bowling_summary.groupby(['match', 'bowlingTeam']).agg({
        'wickets': 'sum',
        'runs': 'sum',
        'economy': 'mean'
    }).reset_index()

    # Prepare match summary with team1 and team2 batting and bowling stats
    match_data = dim_match_summary.copy()

    # Merge batting stats for team1 and team2
    match_data = match_data.merge(
        batting_summary,
        left_on=['match_id', 'team1'],
        right_on=['match', 'teamInnings'],
        how='left',
        suffixes=('', '_team1')
    )
    match_data = match_data.rename(columns={
        'runs': 'team1_runs',
        'balls': 'team1_balls',
        '4s': 'team1_4s',
        '6s': 'team1_6s'
    })

    match_data = match_data.merge(
        batting_summary,
        left_on=['match_id', 'team2'],
        right_on=['match', 'teamInnings'],
        how='left',
        suffixes=('', '_team2')
    )
    match_data = match_data.rename(columns={
        'runs': 'team2_runs',
        'balls': 'team2_balls',
        '4s': 'team2_4s',
        '6s': 'team2_6s'
    })

    # Merge bowling stats for team1 and team2
    match_data = match_data.merge(
        bowling_summary,
        left_on=['match_id', 'team1'],
        right_on=['match', 'bowlingTeam'],
        how='left',
        suffixes=('', '_team1_bowl')
    )
    match_data = match_data.rename(columns={
        'wickets': 'team1_wickets',
        'runs': 'team1_runs_conceded',
        'economy': 'team1_economy'
    })

    match_data = match_data.merge(
        bowling_summary,
        left_on=['match_id', 'team2'],
        right_on=['match', 'bowlingTeam'],
        how='left',
        suffixes=('', '_team2_bowl')
    )
    match_data = match_data.rename(columns={
        'wickets': 'team2_wickets',
        'runs': 'team2_runs_conceded',
        'economy': 'team2_economy'
    })

    # Select relevant columns for prediction
    features = [
        'match_id', 'team1', 'team2', 'ground', 'matchDate',
        'team1_runs', 'team1_balls', 'team1_4s', 'team1_6s',
        'team2_runs', 'team2_balls', 'team2_4s', 'team2_6s',
        'team1_wickets', 'team1_runs_conceded', 'team1_economy',
        'team2_wickets', 'team2_runs_conceded', 'team2_economy',
        'winner', 'margin'
    ]

    match_data = match_data[features]

    return match_data

def preprocess_bowling_performance_data():
    # Merge bowling summary with player details
    bowling_data = fact_bowling_summary.merge(
        dim_players[['name', 'bowlingStyle']],
        left_on='bowlerName',
        right_on='name',
        how='left'
    )
    # Select relevant features and target variables
    features = ['overs', 'maiden', 'wides', 'noBalls', 'bowlingStyle']
    target = ['wickets', 'economy']
    # Identification columns to keep
    id_columns = ['match', 'bowlerName']

    # Select all relevant columns
    all_columns = id_columns + features + target
    bowling_data = bowling_data[all_columns]

    return bowling_data

def preprocess_batting_performance_data():
    # Merge batting summary with player details
    batting_data = fact_bating_summary.merge(
        dim_players[['name', 'battingStyle']],
        left_on='batsmanName',
        right_on='name',
        how='left'
    )
    # Select relevant features and target variables
    features = ['battingPos', 'battingStyle', 'balls', '4s', '6s']
    target = ['runs', 'SR']  # Strike Rate
    # Identification columns to keep
    id_columns = ['match', 'batsmanName']

    # Select all relevant columns
    all_columns = id_columns + features + target
    batting_data = batting_data[all_columns]

    return batting_data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Existing imports and functions remain unchanged above

def train_evaluate_match_outcome_model(data):
    # Encode categorical variables
    le_team1 = LabelEncoder()
    le_team2 = LabelEncoder()
    le_ground = LabelEncoder()
    le_winner = LabelEncoder()

    data = data.dropna(subset=['winner'])  # Drop rows with missing winner

    data['team1_enc'] = le_team1.fit_transform(data['team1'])
    data['team2_enc'] = le_team2.fit_transform(data['team2'])
    data['ground_enc'] = le_ground.fit_transform(data['ground'])
    data['winner_enc'] = le_winner.fit_transform(data['winner'])

    features = [
        'team1_enc', 'team2_enc', 'ground_enc',
        'team1_runs', 'team1_balls', 'team1_4s', 'team1_6s',
        'team2_runs', 'team2_balls', 'team2_4s', 'team2_6s',
        'team1_wickets', 'team1_runs_conceded', 'team1_economy',
        'team2_wickets', 'team2_runs_conceded', 'team2_economy'
    ]

    X = data[features]
    y = data['winner_enc']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Match Outcome Prediction Accuracy: {accuracy:.2f}")

    # Save match outcome predictions before processing margin data
    match_outcome_results = data.iloc[X_test.index].copy()
    match_outcome_results['predicted_winner'] = le_winner.inverse_transform(clf.predict(X_test))
    
    # Regression for margin prediction (simplified: convert margin to numeric)
    margin_data = data.copy()
    print(f"Original margin column count: {margin_data['margin'].notnull().sum()}")
    margin_data['margin_numeric'] = margin_data['margin'].str.extract(r'(\d+)')
    print(f"Extracted margin_numeric count before dropna: {margin_data['margin_numeric'].notnull().sum()}")
    margin_data['margin_numeric'] = pd.to_numeric(margin_data['margin_numeric'], errors='coerce')
    margin_data = margin_data.dropna(subset=['margin_numeric'])
    print(f"Margin_numeric count after dropna: {margin_data['margin_numeric'].notnull().sum()}")

    if margin_data.empty:
        print("No data available for margin prediction after processing. Skipping margin model training.")
        # Save match outcome predictions
        match_outcome_results = match_outcome_results[['match_id', 'team1', 'team2', 'ground', 'matchDate', 'winner', 'predicted_winner', 'margin']]
        match_outcome_results['predicted_margin'] = None
        match_outcome_results.to_excel("Cricket_Dashboard/match_outcome_predictions.xlsx", index=False)
        print("Saved match outcome predictions to Cricket_Dashboard/match_outcome_predictions.xlsx")
        return

    # Reset index to align with train_test_split indices
    margin_data = margin_data.reset_index(drop=True)
    X_margin = margin_data[features]
    y_margin = margin_data['margin_numeric']

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_margin, y_margin, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train_m, y_train_m)
    y_pred_m = reg.predict(X_test_m)

    mse = mean_squared_error(y_test_m, y_pred_m)
    r2 = r2_score(y_test_m, y_pred_m)
    print(f"Margin Prediction MSE: {mse:.2f}, R2: {r2:.2f}")

    # Add margin predictions to match outcome results
    match_outcome_results_with_margin = margin_data.iloc[X_test_m.index].copy()
    match_outcome_results_with_margin['predicted_margin'] = y_pred_m
    # Combine results from both datasets
    match_outcome_results = match_outcome_results.merge(
        match_outcome_results_with_margin[['match_id', 'predicted_margin']], 
        on='match_id', 
        how='left'
    )
    match_outcome_results = match_outcome_results[['match_id', 'team1', 'team2', 'ground', 'matchDate', 'winner', 'predicted_winner', 'margin', 'predicted_margin']]
    match_outcome_results.to_excel("Cricket_Dashboard/match_outcome_predictions.xlsx", index=False)
    print("Saved match outcome and margin predictions to Cricket_Dashboard/match_outcome_predictions.xlsx")

def train_evaluate_batting_performance_model(data):
    # Clean data: replace '-' with NaN and convert numeric columns
    data = data.replace('-', pd.NA)
    numeric_cols = ['battingPos', 'balls', '4s', '6s', 'runs', 'SR']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    print(f"Batting data shape after cleaning: {data.shape}")

    # Reset index to align with train_test_split indices
    data = data.reset_index(drop=True)

    le_battingStyle = LabelEncoder()
    data['battingStyle_enc'] = le_battingStyle.fit_transform(data['battingStyle'])

    features = ['battingPos', 'battingStyle_enc', 'balls', '4s', '6s']
    target_runs = 'runs'
    target_sr = 'SR'

    X = data[features]
    y_runs = data[target_runs]
    y_sr = data[target_sr]

    X_train, X_test, y_train_runs, y_test_runs = train_test_split(X, y_runs, test_size=0.2, random_state=42)
    _, _, y_train_sr, y_test_sr = train_test_split(X, y_sr, test_size=0.2, random_state=42)

    reg_runs = RandomForestRegressor(random_state=42)
    reg_runs.fit(X_train, y_train_runs)
    y_pred_runs = reg_runs.predict(X_test)

    reg_sr = RandomForestRegressor(random_state=42)
    reg_sr.fit(X_train, y_train_sr)
    y_pred_sr = reg_sr.predict(X_test)

    mse_runs = mean_squared_error(y_test_runs, y_pred_runs)
    r2_runs = r2_score(y_test_runs, y_pred_runs)
    mse_sr = mean_squared_error(y_test_sr, y_pred_sr)
    r2_sr = r2_score(y_test_sr, y_pred_sr)

    print(f"Batting Runs Prediction MSE: {mse_runs:.2f}, R2: {r2_runs:.2f}")
    print(f"Batting Strike Rate Prediction MSE: {mse_sr:.2f}, R2: {r2_sr:.2f}")

    # Save batting performance predictions to Excel
    batting_results = data.iloc[X_test.index].copy()
    batting_results['predicted_runs'] = y_pred_runs
    batting_results['predicted_SR'] = y_pred_sr
    # Ensure all relevant columns are included in the output
    columns_to_save = ['match', 'batsmanName', 'battingPos', 'battingStyle', 'balls', '4s', '6s', 'runs', 'SR', 'predicted_runs', 'predicted_SR']
    columns_to_save = [col for col in columns_to_save if col in batting_results.columns]
    batting_results = batting_results[columns_to_save]
    batting_results.to_excel("Cricket_Dashboard/batting_performance_predictions.xlsx", index=False)
    print("Saved batting performance predictions to Cricket_Dashboard/batting_performance_predictions.xlsx")

def train_evaluate_bowling_performance_model(data):
    data = data.dropna()
    # Reset index to align with train_test_split indices
    data = data.reset_index(drop=True)
    
    le_bowlingStyle = LabelEncoder()
    data['bowlingStyle_enc'] = le_bowlingStyle.fit_transform(data['bowlingStyle'])

    features = ['overs', 'maiden', 'wides', 'noBalls', 'bowlingStyle_enc']
    target_wickets = 'wickets'
    target_economy = 'economy'

    X = data[features]
    y_wickets = data[target_wickets]
    y_economy = data[target_economy]

    X_train, X_test, y_train_wickets, y_test_wickets = train_test_split(X, y_wickets, test_size=0.2, random_state=42)
    _, _, y_train_economy, y_test_economy = train_test_split(X, y_economy, test_size=0.2, random_state=42)

    reg_wickets = RandomForestRegressor(random_state=42)
    reg_wickets.fit(X_train, y_train_wickets)
    y_pred_wickets = reg_wickets.predict(X_test)

    reg_economy = RandomForestRegressor(random_state=42)
    reg_economy.fit(X_train, y_train_economy)
    y_pred_economy = reg_economy.predict(X_test)

    mse_wickets = mean_squared_error(y_test_wickets, y_pred_wickets)
    r2_wickets = r2_score(y_test_wickets, y_pred_wickets)
    mse_economy = mean_squared_error(y_test_economy, y_pred_economy)
    r2_economy = r2_score(y_test_economy, y_pred_economy)

    print(f"Bowling Wickets Prediction MSE: {mse_wickets:.2f}, R2: {r2_wickets:.2f}")
    print(f"Bowling Economy Prediction MSE: {mse_economy:.2f}, R2: {r2_economy:.2f}")

    # Save bowling performance predictions to Excel
    bowling_results = data.iloc[X_test.index].copy()
    bowling_results['predicted_wickets'] = y_pred_wickets
    bowling_results['predicted_economy'] = y_pred_economy
    # Ensure all relevant columns are included in the output
    columns_to_save = ['match', 'bowlerName', 'overs', 'maiden', 'wides', 'noBalls', 'bowlingStyle', 'wickets', 'economy', 'predicted_wickets', 'predicted_economy']
    columns_to_save = [col for col in columns_to_save if col in bowling_results.columns]
    bowling_results = bowling_results[columns_to_save]
    bowling_results.to_excel("Cricket_Dashboard/bowling_performance_predictions.xlsx", index=False)
    print("Saved bowling performance predictions to Cricket_Dashboard/bowling_performance_predictions.xlsx")

if __name__ == "__main__":
    load_data()
    match_outcome_data = preprocess_match_outcome_data()
    print("Preprocessed match outcome data sample:")
    # print(match_outcome_data.head())

    batting_data = preprocess_batting_performance_data()
    print("Preprocessed batting performance data sample:")
    # print(batting_data.head())

    bowling_data = preprocess_bowling_performance_data()
    print("Preprocessed bowling performance data sample:")
    # print(bowling_data.head())

    train_evaluate_match_outcome_model(match_outcome_data)
    train_evaluate_batting_performance_model(batting_data)
    train_evaluate_bowling_performance_model(bowling_data)