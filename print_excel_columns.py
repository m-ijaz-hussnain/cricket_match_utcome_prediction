import pandas as pd

files = [
    "Cricket_Dashboard/Cricket_Data/dim_match_summary.xlsx",
    "Cricket_Dashboard/Cricket_Data/dim_players.xlsx",
    "Cricket_Dashboard/Cricket_Data/fact_bating_summary.xlsx",
    "Cricket_Dashboard/Cricket_Data/fact_bowling_summary.xlsx",
    "Cricket_Dashboard\Cricket_Data\match_outcome_predictions.xlsx",
    "Cricket_Dashboard\Cricket_Data\batting_performance_predictions.xlsx",
    "Cricket_Dashboard\Cricket_Data\bowling_performance_predictions.xlsx"
]

for file in files:
    df = pd.read_excel(file)
    print(f"Columns in {file}:")
    print(df.columns.tolist())
    print()