"""
Analyze metrics from evaluation runs.
Computes win rates and survival statistics from metrics.csv files.
"""

import pandas as pd
import argparse
from pathlib import Path
import sys

def analyze_metrics(file_path: str, label: str):
    """Analyze a single metrics file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return None
        
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    total_games = len(df)
    if total_games == 0:
        print(f"No games found in {file_path}")
        return None
        
    # Win rates
    mafia_wins = df[df['winner'] == 'mafia']
    town_wins = df[df['winner'] == 'village']
    
    mafia_win_rate = len(mafia_wins) / total_games
    town_win_rate = len(town_wins) / total_games
    
    # Survival stats
    # Town survived per mafia win
    town_survived_mafia_win = mafia_wins['town_survived'].mean() if len(mafia_wins) > 0 else 0
    
    # Town survived per town win
    town_survived_town_win = town_wins['town_survived'].mean() if len(town_wins) > 0 else 0
    
    # Mafia survived per mafia win
    mafia_survived_mafia_win = mafia_wins['mafia_survived'].mean() if len(mafia_wins) > 0 else 0
    
    # Average game length
    avg_rounds = df['total_rounds'].mean()
    
    print(f"\n=== Analysis for {label} ===")
    print(f"Source: {file_path}")
    print(f"Total Games: {total_games}")
    print(f"Mafia Win Rate: {mafia_win_rate:.2%} ({len(mafia_wins)} wins)")
    print(f"Town Win Rate:  {town_win_rate:.2%} ({len(town_wins)} wins)")
    print(f"Avg Game Length: {avg_rounds:.2f} rounds")
    print("-" * 30)
    print(f"Avg Town Survivors (when Mafia wins): {town_survived_mafia_win:.2f}")
    print(f"Avg Town Survivors (when Town wins):  {town_survived_town_win:.2f}")
    print(f"Avg Mafia Survivors (when Mafia wins): {mafia_survived_mafia_win:.2f}")
    
    return {
        "label": label,
        "mafia_win_rate": mafia_win_rate,
        "town_survived_mafia_win": town_survived_mafia_win,
        "town_survived_town_win": town_survived_town_win
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze Mafia evaluation metrics")
    parser.add_argument("--base", default="eval_games/base_eval/metrics.csv", help="Path to base model metrics")
    parser.add_argument("--public", default="eval_games/public_eval/metrics.csv", help="Path to public CoT metrics")
    parser.add_argument("--private", default="eval_games/private_eval/metrics.csv", help="Path to private CoT metrics (optional)")
    
    args = parser.parse_args()
    
    results = []
    
    # Analyze Base
    res = analyze_metrics(args.base, "Base Model")
    if res: results.append(res)
    
    # Analyze Public
    res = analyze_metrics(args.public, "Public CoT")
    if res: results.append(res)
    
    # Analyze Private (if exists)
    if Path(args.private).exists():
        res = analyze_metrics(args.private, "Private CoT")
        if res: results.append(res)
        
    # Comparison Table
    if results:
        print("\n=== Summary Comparison ===")
        print(f"{'Model':<15} | {'Mafia WR':<10} | {'Town Surv (M Win)':<18} | {'Town Surv (T Win)':<18}")
        print("-" * 70)
        for r in results:
            print(f"{r['label']:<15} | {r['mafia_win_rate']:.2%}     | {r['town_survived_mafia_win']:.2f}               | {r['town_survived_town_win']:.2f}")

if __name__ == "__main__":
    main()
