import argparse
import os 
import concurrent.futures
import json
from itertools import product
import multiprocessing
from typing import List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd 
import numpy as np 


@dataclass 
class Venue:
    ask: float # Best ask 
    ask_sz: int # Best ask size 
    fee: float = 0.0000
    rebate: float = 0.0030


def parse(file_path: str) -> pd.DataFrame:
    """
    Preprocess market data from a CSV file into a cleaned DataFrame with venue information.
    """
    df = pd.read_csv(file_path)
    required_columns = [
        'ts_event', 'publisher_id', 
        'ask_px_00', 'ask_sz_00'
    ]
    assert all(col in df.columns for col in required_columns), \
        f"Missing one or more required columns. Required: {required_columns}"
    assert df['publisher_id'].dtype in [np.int64, np.int32], "publisher_id must be integer type"
    assert df['ask_px_00'].dtype in [np.float64, np.float32], "ask_px_00 must be float type"
    assert df['ask_sz_00'].dtype in [np.float64, np.int64, np.int32], "ask_sz_00 must be numeric type"
    df = df.sort_values(by=['ts_event', 'publisher_id'])
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'])
    df['ts_event'] = pd.to_datetime(df['ts_event'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    df = df[required_columns]
    df['venue'] = df.apply(lambda row: Venue(
        ask=row['ask_px_00'],
        ask_sz=row['ask_sz_00'],
    ), axis=1)
    df = df.groupby('ts_event')['venue'].apply(list).reset_index()
    df = df.set_index('ts_event')
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert all(isinstance(venues, list) for venues in df['venue']), "Venue column must contain lists"
    assert all(all(isinstance(v, Venue) for v in venues) for venues in df['venue']), \
        "All elements in venue lists must be Venue objects"
    return df


def compute_cost(
    split:          List[int],
    venues:         List[Venue],
    order_size:     int,
    lambda_over:    float,
    lambda_under:   float,
    theta_queue:    float,
) -> float:
    executed   = 0
    cash_spent = 0.0

    for i in range(len(venues)):
        exe = min(split[i], venues[i].ask_sz)
        executed   += exe
        cash_spent += exe * (venues[i].ask + venues[i].fee)
        maker_rebate = max(split[i] - exe, 0) * venues[i].rebate
        cash_spent  -= maker_rebate

    underfill = max(order_size - executed, 0)
    overfill  = max(executed - order_size, 0)
    risk_pen  = theta_queue * (underfill + overfill)
    cost_pen  = lambda_under * underfill + lambda_over * overfill

    return cash_spent + risk_pen + cost_pen


def allocate(
    order_size:    int,
    venues:        List[Venue],
    lambda_over:   float,
    lambda_under:  float,
    theta_queue:   float,
) -> Tuple[List[int], float]:
    """
    Returns the split and its cost that minimize total expected cost,
    allowing under- and over-fill penalties for any candidate allocation.
    """
    splits: List[List[int]] = [[]]
    step = 100
    for v in range(len(venues)):
        new_splits: List[List[int]] = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size-used, venues[v].ask_sz)
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost: float = float('inf')
    best_split: List[int] = []

    for alloc in splits:
        if sum(alloc) != order_size: continue
        cost = compute_cost(
            split=alloc,
            venues=venues,
            order_size=order_size,
            lambda_over=lambda_over,
            lambda_under=lambda_under,
            theta_queue=theta_queue
        )
        if cost < best_cost:
            best_cost, best_split = cost, alloc

    if not best_split:
        return None, None

    return best_split, best_cost


def backtest_take_best_ask(
    df: pd.DataFrame,
    lambda_over: float = 0.05,
    lambda_under: float = 0.05,
    theta_queue: float = 0.0005,
) -> float:
    order_size = 5000
    orders_filled = 0
    time_idx = 0
    total_cost = 0.0

    while orders_filled < order_size and time_idx < len(df):
        venue_list = df['venue'].iloc[time_idx]
        remaining = order_size - orders_filled

        # pick best ask
        best_idx = min(range(len(venue_list)), key=lambda i: venue_list[i].ask)
        requested = min(remaining, venue_list[best_idx].ask_sz)

        split = [0] * len(venue_list)
        split[best_idx] = requested

        cost = compute_cost(
            split=split,
            venues=venue_list,
            order_size=requested,
            lambda_over=lambda_over,
            lambda_under=lambda_under,
            theta_queue=theta_queue
        )
        executed = requested
        orders_filled += executed
        total_cost += cost
        time_idx += 1

    return total_cost


def backtest_twap(
    df: pd.DataFrame,
    lambda_over: float = 0.05,
    lambda_under: float = 0.05,
    theta_queue: float = 0.0005,
) -> float:
    order_size = 5000
    orders_filled = 0
    time_idx = 0
    total_cost = 0.0
    bucket_start = df.index[0]
    bucket_prices: List[float] = []
    bucket_venues: List[List[Venue]] = []

    while orders_filled < order_size and time_idx < len(df):
        now = df.index[time_idx]
        if now < bucket_start + pd.Timedelta(seconds=60):
            bucket_venues.append(df['venue'].iloc[time_idx])
            bucket_prices.append(
                min(v.ask for v in bucket_venues[-1])
            )
            time_idx += 1
            continue

        if bucket_prices:
            twap_price = sum(bucket_prices) / len(bucket_prices)
            for venues in bucket_venues:
                if orders_filled >= order_size:
                    break
                sorted_idxs = sorted(
                    range(len(venues)),
                    key=lambda i: abs(venues[i].ask - twap_price)
                )
                for i in sorted_idxs:
                    if orders_filled >= order_size:
                        break
                    v = venues[i]
                    shares_needed = min(
                        order_size - orders_filled,
                        v.ask_sz,
                        max(1, order_size // len(bucket_prices))
                    )
                    if shares_needed <= 0:
                        continue
                    split = [0] * len(venues)
                    split[i] = shares_needed
                    cost = compute_cost(
                        split=split,
                        venues=venues,
                        order_size=shares_needed,
                        lambda_over=lambda_over,
                        lambda_under=lambda_under,
                        theta_queue=theta_queue
                    )
                    orders_filled += shares_needed
                    total_cost += cost
            bucket_prices = []
            bucket_venues = []
            bucket_start = now
        else:
            bucket_start = now
            time_idx += 1

    return total_cost


def backtest_vwap(
    df: pd.DataFrame,
    lambda_over: float = 0.05,
    lambda_under: float = 0.05,
    theta_queue: float = 0.0005,
) -> float:
    order_size = 5000
    orders_filled = 0
    time_idx = 0
    total_cost = 0.0
    bucket_start = df.index[0]
    bucket_prices: List[float] = []
    bucket_sizes: List[int] = []
    bucket_venues: List[List[Venue]] = []

    while orders_filled < order_size and time_idx < len(df):
        now = df.index[time_idx]
        if now < bucket_start + pd.Timedelta(seconds=60):
            venues = df['venue'].iloc[time_idx]
            best = min(range(len(venues)), key=lambda i: venues[i].ask)
            bucket_venues.append(venues)
            bucket_prices.append(venues[best].ask)
            bucket_sizes.append(
                sum(v.ask_sz for v in venues if v.ask == venues[best].ask)
            )
            time_idx += 1
            continue

        if bucket_prices:
            total_vol = sum(bucket_sizes)
            vwap_price = sum(p * s for p, s in zip(bucket_prices, bucket_sizes)) / total_vol
            for venues in bucket_venues:
                if orders_filled >= order_size:
                    break
                sorted_idxs = sorted(
                    range(len(venues)),
                    key=lambda i: (abs(venues[i].ask - vwap_price), -venues[i].ask_sz)
                )
                for i in sorted_idxs:
                    if orders_filled >= order_size:
                        break
                    v = venues[i]
                    weight = v.ask_sz / total_vol
                    shares_needed = min(
                        order_size - orders_filled,
                        v.ask_sz,
                        int(order_size * weight)
                    )
                    if shares_needed <= 0:
                        continue
                    split = [0] * len(venues)
                    split[i] = shares_needed
                    cost = compute_cost(
                        split=split,
                        venues=venues,
                        order_size=shares_needed,
                        lambda_over=lambda_over,
                        lambda_under=lambda_under,
                        theta_queue=theta_queue
                    )
                    orders_filled += shares_needed
                    total_cost += cost
            bucket_prices = []
            bucket_sizes = []
            bucket_venues = []
            bucket_start = now
        else:
            bucket_start = now
            time_idx += 1

    return total_cost


def backtest_contkukanov(
    df: pd.DataFrame,
    lam_under: float = 0.05,
    lam_over: float = 0.05,
    theta_queue: float = 0.0005
) -> float:
    CHUNK = 100
    order_size = 5000
    orders_filled = 0
    time_idx = 0
    total_cost = 0.0

    while orders_filled < order_size and time_idx < len(df):
        venue_list = df['venue'].iloc[time_idx]
        remaining = order_size - orders_filled

        chunk_size = min(remaining, CHUNK)
        split, _ = allocate(
            order_size=chunk_size,
            venues=venue_list,
            lambda_over=lam_over,
            lambda_under=lam_under,
            theta_queue=theta_queue
        )
        if split is None:
            time_idx += 1
            continue
        executed = sum(min(s, v.ask_sz) for s, v in zip(split, venue_list))
        orders_filled += executed
        cost = compute_cost(
            split=split,
            venues=venue_list,
            order_size=remaining,
            lambda_over=lam_over,
            lambda_under=lam_under,
            theta_queue=theta_queue
        )
        total_cost += cost
        time_idx += 1

    return total_cost

def optimize_contkukanov(grid: List[Tuple[float, float, float]], df: pd.DataFrame) -> Tuple[float, Tuple[float, float, float]]:
    best_cost = float('inf')
    best_params = None
    for lam_under, lam_over, theta_queue in grid:
        cost = backtest_contkukanov(df, lam_under, lam_over, theta_queue)
        if cost == 0:
            continue
        elif cost < best_cost:
            best_cost = cost
            best_params = (lam_under, lam_over, theta_queue)
    return best_cost, best_params


def optimize_contkukanov_parallel(grid: List[Tuple[float, float, float]], df: pd.DataFrame) -> Tuple[float, Tuple[float, float, float]]:
    param_grid = list(grid)
    num_cores = multiprocessing.cpu_count()
    max_workers = min(num_cores * 2, len(param_grid))
    print(f"Optimizing using {max_workers} threads across {num_cores} CPU cores")

    def evaluate_params(params: Tuple[float, float, float]) -> Tuple[float, Tuple[float, float, float]]:
        lam_under, lam_over, theta_queue = params
        cost = backtest_contkukanov(df, lam_under, lam_over, theta_queue)
        return cost, params

    best_cost = float('inf')
    best_params = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(evaluate_params, params): params
            for params in param_grid
        }
        for future in concurrent.futures.as_completed(future_to_params):
            cost, params = future.result()
            if cost == 0:
                continue
            elif cost < best_cost:
                best_cost = cost
                best_params = params
    if best_params is None:
        raise RuntimeError("Optimization failed to find valid parameters")
    return best_cost, best_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest the Cont-Kukanov Optimal Order Execution Model')
    parser.add_argument('--backtest_file', type=str, default='l1_day.csv', help='Path to the CSV file containing market data')
    parser.add_argument('--parallel', action='store_true', help='Run optimization in parallel')
    args = parser.parse_args()

    print(f"Parsing data from {args.backtest_file}\n")
    df = parse(args.backtest_file)

    print(f"Default risk parameters from paper: \
          \nlambda_under=0.05, \
          \nlambda_over=0.05, \
          \ntheta_queue=0.0005\n")
    print(f"\nBaseline risk parameters results:")
    print(f'Best Ask Total Cost: {backtest_take_best_ask(df):.2f}')
    print(f'TWAP Total Cost: {backtest_twap(df):.2f}')
    print(f'VWAP Total Cost: {backtest_vwap(df):.2f}')
    print(f'Cont-Kukanov Total Cost: {backtest_contkukanov(df):.2f}\n')

    param = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    grid = product(param, param, param)

    print(f"Optimizing risk parameters over uniform grid: {[float(x) for x in param]}")
    if args.parallel:
        best_cost, (best_lam_under, best_lam_over, best_theta_queue) = optimize_contkukanov_parallel(grid, df)
    else:
        best_cost, (best_lam_under, best_lam_over, best_theta_queue) = optimize_contkukanov(grid, df)
    print(f'Best Cost: {best_cost:.1f}')
    print(f'Best Parameters:')
    print(f'  lambda_under: {best_lam_under:.6f}')
    print(f'  lambda_over:  {best_lam_over:.6f}')
    print(f'  theta_queue:  {best_theta_queue:.6f}')
