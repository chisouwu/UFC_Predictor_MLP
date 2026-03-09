import os
import argparse
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TabularUFCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_features(df: pd.DataFrame):
    # Keep only pre-fight features (no in-fight / per-fight aggregated stats)
    allowed_suffixes = {
        'wins', 'losses', 'draws',
        'height', 'weight', 'reach',
    }

    # select r_ and b_ columns that are pre-fight
    r_cols = [c for c in df.columns if c.startswith('r_') and c[2:] in allowed_suffixes]
    b_cols = [c for c in df.columns if c.startswith('b_') and c[2:] in allowed_suffixes]

    # ensure matching suffixes on both sides
    r_suffixes = {c[2:]: c for c in r_cols}
    b_suffixes = {c[2:]: c for c in b_cols}
    common = sorted(set(r_suffixes) & set(b_suffixes))

    r_sel = [r_suffixes[s] for s in common]
    b_sel = [b_suffixes[s] for s in common]

    # numeric values, fill NaN
    r_vals = df[r_sel].astype(float).fillna(0.0) if r_sel else pd.DataFrame(np.zeros((len(df), 0)))
    b_vals = df[b_sel].astype(float).fillna(0.0) if b_sel else pd.DataFrame(np.zeros((len(df), 0)))

    # difference features (red - blue)
    X_diff = (r_vals.values - b_vals.values)

    # simple categorical: division factorized
    div, div_idx = pd.factorize(df['division'].fillna('unknown'))
    div = div.reshape(-1, 1).astype(float)

    # misc numeric pre-fight features
    misc = df[['title_fight', 'total_rounds']].fillna(0.0).astype(float).values

    X = np.hstack([X_diff, div, misc])
    return X, div_idx


def prepare_datasets(csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path, parse_dates=['date'], dayfirst=False)

    # drop rows missing winner info
    df = df.dropna(subset=['winner_id'])

    # sort chronologically
    df = df.sort_values('date').reset_index(drop=True)

    # build time-aware features: for each fight, compute fighter history using only earlier fights
    def build_time_aware_features(df):
        # history per fighter
        history = {}
        rows = []
        feat_names = []

        # helper to get history summary for a fighter
        def get_hist(fid):
            h = history.get(fid)
            if h is None:
                return {
                    'fights': 0, 'wins': 0, 'losses': 0, 'draws': 0,
                    'total_sig_str_landed': 0.0, 'total_sig_str_atmpted': 0.0,
                    'total_td_landed': 0.0, 'total_match_min': 0.0,
                    'total_sub_att': 0.0
                }
            return h

        for _, row in df.iterrows():
            r_id = row.get('r_id')
            b_id = row.get('b_id')

            rh = get_hist(r_id)
            bh = get_hist(b_id)

            # compute per-fighter derived pre-fight features from history
            def summarize(h):
                fights = h['fights']
                if fights == 0 or fights is None or np.isnan(fights):
                    avg_sig_landed = 0.0
                    avg_sig_atmpt = 0.0
                    avg_td = 0.0
                    avg_min = 0.0
                    avg_sub_att = 0.0
                    win_rate = 0.0
                else:
                    avg_sig_landed = h['total_sig_str_landed'] / fights if fights else 0.0
                    avg_sig_atmpt = h['total_sig_str_atmpted'] / fights if fights else 0.0
                    avg_td = h['total_td_landed'] / fights if fights else 0.0
                    avg_min = h['total_match_min'] / fights if fights else 0.0
                    avg_sub_att = h['total_sub_att'] / fights if fights else 0.0
                    win_rate = h['wins'] / fights if fights else 0.0
                return [fights, h['wins'], h['losses'], h['draws'], avg_sig_landed, avg_sig_atmpt, avg_td, avg_sub_att, avg_min, win_rate]

            r_feats = summarize(rh)
            b_feats = summarize(bh)

            # static pre-fight attributes (height/weight/reach) taken from current row
            r_static = [row.get('r_height', 0.0) or 0.0, row.get('r_weight', 0.0) or 0.0, row.get('r_reach', 0.0) or 0.0]
            b_static = [row.get('b_height', 0.0) or 0.0, row.get('b_weight', 0.0) or 0.0, row.get('b_reach', 0.0) or 0.0]

            # division, title_fight, total_rounds
            div = row.get('division')
            title = row.get('title_fight', 0.0) or 0.0
            rounds = row.get('total_rounds', 0.0) or 0.0

            feat_row = r_feats + r_static + b_feats + b_static + [0.0 if pd.isna(div) else div, title, rounds]
            rows.append(feat_row)

            # update history with this fight's in-fight results for future fights
            # red fighter
            r_h = history.setdefault(r_id, {'fights':0,'wins':0,'losses':0,'draws':0,'total_sig_str_landed':0.0,'total_sig_str_atmpted':0.0,'total_td_landed':0.0,'total_match_min':0.0,'total_sub_att':0.0})
            b_h = history.setdefault(b_id, {'fights':0,'wins':0,'losses':0,'draws':0,'total_sig_str_landed':0.0,'total_sig_str_atmpted':0.0,'total_td_landed':0.0,'total_match_min':0.0,'total_sub_att':0.0})

            # outcome
            winner = row.get('winner_id')
            if pd.notna(winner):
                if winner == r_id:
                    r_h['wins'] += 1
                    b_h['losses'] += 1
                elif winner == b_id:
                    b_h['wins'] += 1
                    r_h['losses'] += 1
                else:
                    r_h['draws'] += 1
                    b_h['draws'] += 1

            # in-fight numeric updates (use post-fight numbers)
            # match time in minutes
            try:
                match_min = float(row.get('match_time_sec', 0.0)) / 60.0
            except Exception:
                match_min = 0.0

            for side, h in (('r_', r_h), ('b_', b_h)):
                try:
                    sig_l = float(row.get(side+'total_str_landed', 0.0) or 0.0)
                except Exception:
                    sig_l = 0.0
                try:
                    sig_a = float(row.get(side+'total_str_atmpted', 0.0) or 0.0)
                except Exception:
                    sig_a = 0.0
                try:
                    td_l = float(row.get(side+'td_landed', 0.0) or 0.0)
                except Exception:
                    td_l = 0.0
                try:
                    sub_a = float(row.get(side+'sub_att', 0.0) or 0.0)
                except Exception:
                    sub_a = 0.0

                h['fights'] += 1
                h['total_sig_str_landed'] += sig_l
                h['total_sig_str_atmpted'] += sig_a
                h['total_td_landed'] += td_l
                h['total_sub_att'] += sub_a
                h['total_match_min'] += match_min

        # convert rows to array and factorize division separately later
        X = np.array(rows, dtype=object)
        return X, history

    X_raw, history = build_time_aware_features(df)

    # division column is at last-2 position in our construction; extract and factorize
    div_col = X_raw[:, -3]
    # convert division factor to numeric codes
    div_codes, div_idx = pd.factorize(div_col)

    # rest numeric features
    X_numeric = X_raw[:, :-3].astype(float)
    title = X_raw[:, -2].astype(float)
    rounds = X_raw[:, -1].astype(float)

    X = np.hstack([X_numeric, div_codes.reshape(-1,1).astype(float), title.reshape(-1,1), rounds.reshape(-1,1)])

    # label: red win
    y = (df['winner_id'] == df['r_id']).astype(int).values

    # temporal split: use earliest train_frac fraction as training, rest as test
    train_frac = 1.0 - test_size
    n = len(df)
    split_idx = int(n * train_frac)
    if split_idx < 1:
        split_idx = 1
    if split_idx >= n:
        split_idx = n-1

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    # Ensure no NaN/Inf after scaling
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    train_ds = TabularUFCDataset(X_train, y_train)
    val_ds = TabularUFCDataset(X_val, y_val)
    return train_ds, val_ds, X_train.shape[1], scaler, div_idx


def train(csv_path, epochs=20, batch_size=32, lr=1e-3, out_path='models/ufc_mlp.pth'):
    train_ds, val_ds, input_dim, scaler, div_idx = prepare_datasets(csv_path)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMLP(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Check for NaN/Inf in input
            if torch.isnan(xb).any() or torch.isinf(xb).any():
                print("[Warning] NaN or Inf detected in input features!")
            if torch.isnan(yb).any() or torch.isinf(yb).any():
                print("[Warning] NaN or Inf detected in labels!")
            logits = model(xb)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("[Warning] NaN or Inf detected in logits!")
            loss = criterion(logits, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN or Inf detected in loss!")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch}/{epochs}  train_loss={avg_loss:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state_dict': model.state_dict(), 'input_dim': input_dim}, out_path)
            with open(out_path + '.scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

    print(f"Best val acc: {best_acc:.4f}. Model saved to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='/teamspace/studios/this_studio/data/UFC.csv')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='models/ufc_mlp.pth')
    args = p.parse_args()

    train(args.csv, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, out_path=args.out)


if __name__ == '__main__':
    main()
