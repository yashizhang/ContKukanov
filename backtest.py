import os 
from datetime import datetime 
import pandas as pd 
import numpy as np 
import json

def allocate(order_size, venues, lam_over, lam_under, theta_queue):
    step = 100
    splits = [[]]
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v]['ask_size'])
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits
    best_cost = float('inf')
    best_split = []
    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc, venues, order_size, lam_over, lam_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc
    return best_split, best_cost

def compute_cost(split, venues, order_size, lam_over, lam_under, theta_queue):
    executed = 0
    cash_spent = 0.0
    for i in range(len(venues)):
        exe = min(split[i], venues[i]['ask_size'])
        executed += exe
        cash_spent += exe * (venues[i]['ask'] + venues[i].get('fee', 0))
        maker_rebate = max(split[i] - exe, 0) * venues[i].get('rebate', 0)
        cash_spent -= maker_rebate
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta_queue * (underfill + overfill)
    cost_pen = lam_under * underfill + lam_over * overfill
    return cash_spent + risk_pen + cost_pen

def simulate_static_strategy(initial_split, snapshots, publisher_ids):
    # Ensure initial_split matches number of venues to avoid index errors
    if len(initial_split) != len(publisher_ids):
        initial_split = [0] * len(publisher_ids)
    remain = initial_split.copy()
    executed = 0
    cash_spent = 0.0
    pid_to_index = {pid: i for i, pid in enumerate(publisher_ids)}
    for snap in snapshots:
        for _, row in snap.iterrows():
            pid = row['publisher_id']
            idx = pid_to_index[pid]
            if remain[idx] > 0:
                fill = min(remain[idx], row['ask_sz_00'])
                if fill > 0:
                    remain[idx] -= fill
                    executed += fill
                    cash_spent += fill * row['ask_px_00']
        if executed >= sum(initial_split):
            break
    return executed, cash_spent

def simulate_best_ask(snapshots, order_size):
    resid = order_size
    executed = 0
    cash_spent = 0.0
    for snap in snapshots:
        df_sorted = snap.sort_values('ask_px_00')
        for _, row in df_sorted.iterrows():
            to_fill = min(resid, row['ask_sz_00'])
            if to_fill > 0:
                executed += to_fill
                cash_spent += to_fill * row['ask_px_00']
                resid -= to_fill
            if resid <= 0:
                break
        if resid <= 0:
            break
    return executed, cash_spent

def simulate_twap(snapshots, order_size):
    first_ts = snapshots[0]['ts_event'].iloc[0]
    buckets = {}
    for snap in snapshots:
        # compute minute bucket index
        diff = snap['ts_event'].iloc[0] - first_ts
        diff_seconds = diff / pd.Timedelta(seconds=1)
        b = int(diff_seconds // 60)
        if b < 9 and b not in buckets:
            buckets[b] = snap
    bucket_indices = sorted(buckets.keys())
    remain = order_size
    executed = 0
    cash_spent = 0.0
    for i, b in enumerate(bucket_indices):
        snap = buckets[b]
        buckets_left = len(bucket_indices) - i
        if i < len(bucket_indices) - 1:
            chunk = remain // buckets_left
        else:
            chunk = remain
        df_sorted = snap.sort_values('ask_px_00')
        to_fill = chunk
        for _, row in df_sorted.iterrows():
            fill = min(to_fill, row['ask_sz_00'])
            if fill > 0:
                executed += fill
                cash_spent += fill * row['ask_px_00']
                remain -= fill
                to_fill -= fill
            if to_fill <= 0:
                break
    return executed, cash_spent

def simulate_vwap(snapshots, order_size, publisher_ids):
    snap0 = snapshots[0]
    pid_to_row = {row['publisher_id']: row for _, row in snap0.iterrows()}
    total_sz = sum(row['ask_sz_00'] for row in pid_to_row.values())
    weights = [pid_to_row.get(pid, {'ask_sz_00': 0})['ask_sz_00'] / total_sz if total_sz > 0 else 0
               for pid in publisher_ids]
    initial_split = [int(order_size * w) for w in weights[:-1]]
    initial_split.append(order_size - sum(initial_split))
    return simulate_static_strategy(initial_split, snapshots, publisher_ids)

def main():
    df = pd.read_csv('l1_day.csv', usecols=['publisher_id', 'ts_event', 'ask_px_00', 'ask_sz_00'])
    # Convert event timestamps to datetime for time calculations
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df.sort_values(['ts_event', 'publisher_id'], inplace=True)
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')
    unique_ts = sorted(df['ts_event'].unique())
    snapshots = [df[df['ts_event'] == ts] for ts in unique_ts]
    publisher_ids = sorted(df['publisher_id'].unique())

    # Prepare initial venues snapshot for parameter search
    initial_snap = snapshots[0]
    venues = []
    for pid in publisher_ids:
        row = initial_snap[initial_snap['publisher_id'] == pid]
        if not row.empty:
            ask = float(row['ask_px_00'].iloc[0])
            ask_sz = int(row['ask_sz_00'].iloc[0])
        else:
            ask, ask_sz = float('inf'), 0
        venues.append({'ask': ask, 'ask_size': ask_sz, 'fee': 0.0, 'rebate': 0.0})

    order_size = 5000
    lam_over_list = [0.0, 0.1, 0.5, 1.0]
    lam_under_list = [0.0, 0.1, 0.5, 1.0]
    theta_list = [0.0, 0.1, 0.5, 1.0]

    # Tune parameters by dynamic simulation cost (cash + penalties for underfill)
    best_cost = float('inf')
    best_params = (0.0, 0.0, 0.0)
    for lam_over in lam_over_list:
        for lam_under in lam_under_list:
            for theta in theta_list:
                # compute static allocation based on cost model
                split, _ = allocate(order_size, venues, lam_over, lam_under, theta)
                # simulate execution dynamically
                executed_sim, cash_sim = simulate_static_strategy(split, snapshots, publisher_ids)
                underfill = max(order_size - executed_sim, 0)
                # compute total cost including penalty for underfill
                total_cost = cash_sim + lam_under * underfill + theta * underfill
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_params = (lam_over, lam_under, theta)
    lam_over, lam_under, theta = best_params

    # Router strategy
    best_split, _ = allocate(order_size, venues, lam_over, lam_under, theta)
    executed_router, cash_router = simulate_static_strategy(best_split, snapshots, publisher_ids)
    avg_router = cash_router / executed_router if executed_router > 0 else None

    # Baselines
    executed_bestask, cash_bestask = simulate_best_ask(snapshots, order_size)
    avg_bestask = cash_bestask / executed_bestask if executed_bestask > 0 else None

    executed_twap, cash_twap = simulate_twap(snapshots, order_size)
    avg_twap = cash_twap / executed_twap if executed_twap > 0 else None

    executed_vwap, cash_vwap = simulate_vwap(snapshots, order_size, publisher_ids)
    avg_vwap = cash_vwap / executed_vwap if executed_vwap > 0 else None

    savings = {
        'best_ask': (avg_bestask - avg_router) / avg_bestask * 10000 if avg_bestask and avg_router else None,
        'twap': (avg_twap - avg_router) / avg_twap * 10000 if avg_twap and avg_router else None,
        'vwap': (avg_vwap - avg_router) / avg_vwap * 10000 if avg_vwap and avg_router else None
    }

    result = {
        'best_parameters': {
            'lambda_over': lam_over,
            'lambda_under': lam_under,
            'theta_queue': theta
        },
        'router': {'total_cash_spent': cash_router, 'avg_price': avg_router},
        'best_ask': {'total_cash_spent': cash_bestask, 'avg_price': avg_bestask},
        'twap': {'total_cash_spent': cash_twap, 'avg_price': avg_twap},
        'vwap': {'total_cash_spent': cash_vwap, 'avg_price': avg_vwap},
        'savings_bps': savings
    }

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
