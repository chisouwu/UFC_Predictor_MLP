import pandas as pd
import torch
import pickle
import numpy as np
from train_mlp import SimpleMLP

FIGHTER_CSV = "data/fighter_details.csv"
MODEL_PATH = "models/ufc_mlp.pth"
SCALER_PATH = "models/ufc_mlp.pth.scaler.pkl"

"""
Usage:
    python predict_hypothetical.py "Name1" "Name2"
    (No references to data/eval, only uses main data files)
"""
def get_fighter_row(name, df):
    matches = df[df['name'].str.lower() == name.lower()]
    if matches.empty:
        return None
    return matches.iloc[0]

def main(name1, name2):
    fighters = pd.read_csv(FIGHTER_CSV)
    red = get_fighter_row(name1, fighters)
    blue = get_fighter_row(name2, fighters)
    if red is None or blue is None:
        print(f"Could not find both fighters: {name1}, {name2}")
        return

    # Model expects 29 features: [red_feats, red_static, blue_feats, blue_static, div_code, title_fight, total_rounds]
    # Static features from fighter_details.csv
    def get_static_feats(row):
        return [
            float(row.get('height', 0) or 0),
            float(row.get('weight', 0) or 0),
            float(row.get('reach', 0) or 0)
        ]

    # Pre-fight stats
    stat_keys = ['wins', 'losses', 'draws', 'splm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_avg_acc', 'td_def', 'sub_avg']
    def get_stats(row):
        return [float(row.get(k, 0) or 0) for k in stat_keys]

    # Build full feature vector
    r_feats = get_stats(red)
    r_static = get_static_feats(red)
    b_feats = get_stats(blue)
    b_static = get_static_feats(blue)

    # Division, title_fight, total_rounds (default values)
    div_code = 0.0
    title_fight = 0.0
    total_rounds = 3.0

    # Assemble features: [r_feats, r_static, b_feats, b_static, div_code, title_fight, total_rounds]
    feats = r_feats + r_static + b_feats + b_static + [div_code, title_fight, total_rounds]
    # Fill missing features to match input_dim
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    input_dim = checkpoint['input_dim']
    if len(feats) < input_dim:
        feats += [0.0] * (input_dim - len(feats))
    elif len(feats) > input_dim:
        feats = feats[:input_dim]
    feats = np.array(feats).reshape(1, -1)

    # Load scaler and model
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    feats_scaled = scaler.transform(feats)
    feats_scaled = np.nan_to_num(feats_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.from_numpy(feats_scaled).float()
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    input_dim = checkpoint['input_dim']
    model = SimpleMLP(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logits = model(x)
    pred = logits.argmax(dim=1).item()
    winner = name1 if pred == 1 else name2
    print(f"Prediction: {winner} is predicted to win between {name1} and {name2}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict_hypothetical.py 'Name1' 'Name2'")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
