import pandas as pd
import torch
import pickle
import numpy as np
import argparse
from train_mlp import SimpleMLP

FIGHTER_CSV = "data/fighter_details.csv"
MODEL_PATH = "models/ufc_mlp.pth"
SCALER_PATH = "models/ufc_mlp.pth.scaler.pkl"

"""Predict winner for a hypothetical matchup.

Usage example:
    python predict_hypothetical.py "Name1" "Name2" \
      --red-close-min -150 --red-close-max -130 \
      --blue-close-min 130 --blue-close-max 150
"""
def get_fighter_row(name, df):
    matches = df[df['name'].str.lower() == name.lower()]
    if matches.empty:
        return None
    return matches.iloc[0]


def safe_float(v, default=0.0):
    try:

        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def compute_close_avg(close_avg, close_min, close_max):
    if close_avg is not None:
        return float(close_avg)
    vals = []
    if close_min is not None:
        vals.append(float(close_min))
    if close_max is not None:
        vals.append(float(close_max))
    return float(np.mean(vals)) if vals else 100.0


def main(name1, name2, red_close_avg=None, red_close_min=None, red_close_max=None, blue_close_avg=None, blue_close_min=None, blue_close_max=None, division_code=0.0, title_fight=0.0, total_rounds=3.0):
    fighters = pd.read_csv(FIGHTER_CSV)
    red = get_fighter_row(name1, fighters)
    blue = get_fighter_row(name2, fighters)
    if red is None or blue is None:
        print(f"Could not find both fighters: {name1}, {name2}")
        return

    # Model feature layout:
    # [r_feats(10), r_static(3), r_close_avg(1), b_feats(10), b_static(3), b_close_avg(1), div_code, title_fight, total_rounds]
    # Static features from fighter_details.csv
    def get_static_feats(row):
        return [
            safe_float(row.get('height', 0), 0.0),
            safe_float(row.get('weight', 0), 0.0),
            safe_float(row.get('reach', 0), 0.0),
        ]

    # Approximate the training-time history features from fighter_details stats.
    def get_hist_like_feats(row):
        wins = safe_float(row.get('wins', 0), 0.0)
        losses = safe_float(row.get('losses', 0), 0.0)
        draws = safe_float(row.get('draws', 0), 0.0)
        fights = wins + losses + draws

        splm = safe_float(row.get('splm', 0), 0.0)
        str_acc = safe_float(row.get('str_acc', 0), 0.0)
        # Some sources store accuracy as percent (e.g., 55). Normalize if needed.
        str_acc = str_acc / 100.0 if str_acc > 1.0 else str_acc
        avg_sig_landed = splm
        avg_sig_atmpt = splm / str_acc if str_acc > 0 else 0.0
        avg_td = safe_float(row.get('td_avg', 0), 0.0)
        avg_sub_att = safe_float(row.get('sub_avg', 0), 0.0)
        avg_min = 15.0
        win_rate = (wins / fights) if fights > 0 else 0.0

        return [
            fights,
            wins,
            losses,
            draws,
            avg_sig_landed,
            avg_sig_atmpt,
            avg_td,
            avg_sub_att,
            avg_min,
            win_rate,
        ]

    # Build full feature vector
    r_feats = get_hist_like_feats(red)
    r_static = get_static_feats(red)
    b_feats = get_hist_like_feats(blue)
    b_static = get_static_feats(blue)

    r_close = compute_close_avg(red_close_avg, red_close_min, red_close_max)
    b_close = compute_close_avg(blue_close_avg, blue_close_min, blue_close_max)

    # Assemble features to match train_mlp.py order.
    feats = r_feats + r_static + [r_close] + b_feats + b_static + [b_close] + [
        float(division_code),
        float(title_fight),
        float(total_rounds),
    ]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("name1", help="Red corner fighter name")
    parser.add_argument("name2", help="Blue corner fighter name")

    # Odds input options: pass average directly, or pass min/max and average is computed.
    parser.add_argument("--red-close-avg", type=float, default=None)
    parser.add_argument("--red-close-min", type=float, default=None)
    parser.add_argument("--red-close-max", type=float, default=None)
    parser.add_argument("--blue-close-avg", type=float, default=None)
    parser.add_argument("--blue-close-min", type=float, default=None)
    parser.add_argument("--blue-close-max", type=float, default=None)

    parser.add_argument("--division-code", type=float, default=0.0)
    parser.add_argument("--title-fight", type=float, default=0.0)
    parser.add_argument("--total-rounds", type=float, default=3.0)

    args = parser.parse_args()
    main(
        args.name1,
        args.name2,
        red_close_avg=args.red_close_avg,
        red_close_min=args.red_close_min,
        red_close_max=args.red_close_max,
        blue_close_avg=args.blue_close_avg,
        blue_close_min=args.blue_close_min,
        blue_close_max=args.blue_close_max,
        division_code=args.division_code,
        title_fight=args.title_fight,
        total_rounds=args.total_rounds,
    )
