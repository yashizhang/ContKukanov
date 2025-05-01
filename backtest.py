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
    
    Args:
        file_path (str): Path to the CSV file containing market data
        
    Returns:
        pd.DataFrame: Cleaned DataFrame indexed by ts_event containing venue information
        
    Raises:
        AssertionError: If required columns are missing or data structure is invalid
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Validate required columns exist
    required_columns = [
        'ts_event', 'publisher_id', 
        'ask_px_00', 'ask_sz_00',
        'bid_px_00', 'bid_sz_00'
    ]
    assert all(col in df.columns for col in required_columns), \
        f"Missing one or more required columns. Required: {required_columns}"
    
    # Validate data types
    assert df['publisher_id'].dtype in [np.int64, np.int32], "publisher_id must be integer type"
    assert df['ask_px_00'].dtype in [np.float64, np.float32], "ask_px_00 must be float type"
    assert df['ask_sz_00'].dtype in [np.float64, np.int64, np.int32], "ask_sz_00 must be numeric type"
    
    # Sort and drop duplicates
    df = df.sort_values(by=['ts_event', 'publisher_id'])
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'])
    
    # Convert timestamp to datetime
    df['ts_event'] = pd.to_datetime(df['ts_event'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    
    # Select relevant columns
    df = df[required_columns]
    
    # Create Venue objects
    df['venue'] = df.apply(lambda row: Venue(
        ask=row['ask_px_00'],
        ask_sz=row['ask_sz_00'],
        fee=0.0000,
        rebate=0.0030
    ), axis=1)
    
    # Group by ts_event and aggregate venues into a list
    df = df.groupby('ts_event')['venue'].apply(list).reset_index()
    df = df.set_index('ts_event')
    
    # Validate output structure
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

    for i, v in enumerate(venues):
        exe = min(split[i], v.ask_sz)
        executed   += exe
        cash_spent += exe * (v.ask + v.fee)
        maker_rebate = max(split[i] - exe, 0) * v.rebate
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
    step:          int = 100, 
) -> Tuple[Optional[List[int]], Optional[float]]:
    """
    Returns (split, cost) if an exact order_size split is found in 'step'-share chunks,
    or (None, None) otherwise — so the caller can skip to the next timestamp.
    """
    splits: List[List[int]] = [[]]
    for v in range(len(venues)):
        new_splits: List[List[int]] = []
        for alloc in splits:
            used  = sum(alloc)
            max_v = min(order_size - used, venues[v].ask_sz)
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits

    best_cost: float = float('inf')
    best_split: List[int] = []

    for alloc in splits:
        if sum(alloc) != order_size:
            continue
        cost = compute_cost(alloc,
                            venues = venues,
                            order_size = order_size,
                            lambda_over = lambda_over,
                            lambda_under = lambda_under,
                            theta_queue = theta_queue)
        if cost < best_cost:
            best_cost, best_split = cost, alloc

    # if we never found exactly order_size, return (None, None)
    if not best_split:
        return None, None

    return best_split, best_cost

def backtest_take_best_ask(df: pd.DataFrame) -> float:
    """
    Implement a simple strategy that takes the best ask price available at each tick.
    This is an aggressive strategy that immediately executes against the best offer
    until the entire order is filled.
    
    Args:
        df: DataFrame with market data indexed by timestamp
        
    Returns:
        float: Total cost of execution
    """
    order_size = 5000  # Total shares to buy
    orders_filled = 0  # Running count of shares filled
    time_idx = 0      # Current position in the market data
    total_cost = 0    # Accumulator for total execution cost
    
    while orders_filled < order_size and time_idx < len(df): 
        # Get the current list of venues
        venue_list = df['venue'].iloc[time_idx]
        
        # Find the venue with the best (lowest) ask price
        best_ask = min(venue_list, key=lambda x: x.ask)

        # Calculate shares to execute at this venue
        shares_to_execute = min(order_size - orders_filled, best_ask.ask_sz)
        
        # Calculate direct execution cost including fees
        execution_cost = shares_to_execute * (best_ask.ask + best_ask.fee)
        
        # Update running totals
        orders_filled += shares_to_execute
        total_cost += execution_cost

        # Move to next tick of market data
        time_idx += 1
        
    return total_cost

def backtest_twap(df: pd.DataFrame) -> float:
    """
    Implement a 60-second TWAP strategy to buy 5000 shares.
    
    Args:
        df: DataFrame with market data indexed by timestamp
        
    Returns:
        float: Total cost of execution
    """
    order_size = 5000
    orders_filled = 0
    time_idx = 0
    total_cost = 0
    bucket_start_time = df.index[0]
    
    # Lists to store prices and venues for TWAP calculation
    bucket_prices = []
    bucket_venues = []
    
    while orders_filled < order_size and time_idx < len(df):
        current_time = df.index[time_idx]
        
        # If we're still within the current 60-second bucket, collect prices
        if current_time < bucket_start_time + pd.Timedelta(seconds=60):
            venue_list = df['venue'].iloc[time_idx]
            bucket_venues.append(venue_list)
            bucket_prices.append(min(v.ask for v in venue_list))
            time_idx += 1
            continue
            
        # Once we have a full 60-second bucket, execute trades based on TWAP
        if bucket_prices:
            # Calculate TWAP price for the bucket
            twap_price = sum(bucket_prices) / len(bucket_prices)
            
            # Find venues with asks close to TWAP price
            for venues in bucket_venues:
                if orders_filled >= order_size:
                    break
                    
                # Sort venues by how close their ask is to TWAP price
                sorted_venues = sorted(venues, key=lambda v: abs(v.ask - twap_price))
                
                # Execute trades at venues closest to TWAP
                for venue in sorted_venues:
                    if orders_filled >= order_size:
                        break
                        
                    # Calculate how many shares to buy from this venue
                    shares_needed = min(
                        order_size - orders_filled,  # Shares still needed
                        venue.ask_sz,  # Available size at venue
                        order_size // len(bucket_prices)  # Roughly equal distribution across bucket
                    )
                    
                    if shares_needed > 0:
                        # Calculate direct execution cost including fees
                        execution_cost = shares_needed * (venue.ask + venue.fee)
                        orders_filled += shares_needed
                        total_cost += execution_cost
            
            # Reset for next bucket
            bucket_prices = []
            bucket_venues = []
            bucket_start_time = current_time
        else:
            # Move to next timestamp if bucket was empty
            time_idx += 1
            bucket_start_time = current_time
            
    return total_cost

def backtest_vwap(df: pd.DataFrame) -> float:
    """
    Implement a VWAP strategy to buy 5000 shares, weighting prices by displayed ask size.
    
    Args:
        df: DataFrame with market data indexed by timestamp
        
    Returns:
        float: Total cost of execution
    """
    order_size = 5000
    orders_filled = 0
    time_idx = 0
    total_cost = 0
    bucket_start_time = df.index[0]
    
    # Lists to store prices, sizes and venues for VWAP calculation
    bucket_prices = []
    bucket_sizes = []
    bucket_venues = []
    
    while orders_filled < order_size and time_idx < len(df):
        current_time = df.index[time_idx]
        
        # If we're still within the current 60-second bucket, collect data
        if current_time < bucket_start_time + pd.Timedelta(seconds=60):
            venue_list = df['venue'].iloc[time_idx]
            
            # Find best ask and its size across venues
            best_ask = min(venue_list, key=lambda x: x.ask)
            total_size_at_best = sum(v.ask_sz for v in venue_list if v.ask == best_ask.ask)
            
            bucket_venues.append(venue_list)
            bucket_prices.append(best_ask.ask)
            bucket_sizes.append(total_size_at_best)
            
            time_idx += 1
            continue
            
        # Once we have a full 60-second bucket, execute trades based on VWAP
        if bucket_prices:
            # Calculate VWAP price for the bucket
            total_volume = sum(bucket_sizes)
            vwap_price = sum(p * s for p, s in zip(bucket_prices, bucket_sizes)) / total_volume
            
            # Find venues with asks close to VWAP price
            for venues in bucket_venues:
                if orders_filled >= order_size:
                    break
                    
                # Sort venues by how close their ask is to VWAP price
                # and by their size (prefer larger sizes for same price)
                sorted_venues = sorted(venues, 
                                    key=lambda v: (abs(v.ask - vwap_price), -v.ask_sz))
                
                # Execute trades at venues closest to VWAP
                for venue in sorted_venues:
                    if orders_filled >= order_size:
                        break
                        
                    # Calculate how many shares to buy from this venue
                    # Weight by relative size compared to total volume in bucket
                    venue_weight = venue.ask_sz / total_volume
                    shares_needed = min(
                        order_size - orders_filled,  # Shares still needed
                        venue.ask_sz,  # Available size at venue
                        int(order_size * venue_weight)  # Size-weighted allocation
                    )
                    
                    if shares_needed > 0:
                        # Calculate direct execution cost including fees
                        execution_cost = shares_needed * (venue.ask + venue.fee)
                        orders_filled += shares_needed
                        total_cost += execution_cost
            
            # Reset for next bucket
            bucket_prices = []
            bucket_sizes = []
            bucket_venues = []
            bucket_start_time = current_time
        else:
            # Move to next timestamp if bucket was empty
            time_idx += 1
            bucket_start_time = current_time
            
    return total_cost

def backtest_contkukanov(df: pd.DataFrame, 
                        lam_under: float = 0.05,
                        lam_over: float = 0.05,
                        theta_queue: float = 0.0005) -> float:
    """
    Implement the Cont-Kukanov model to buy 5000 shares using optimal allocation.
    This strategy uses the allocate() function to determine optimal order splits
    across venues at each tick.
    
    Args:
        df: DataFrame with market data indexed by timestamp
        lam_under: Cost penalty per unfilled share (default: 0.05)
        lam_over: Cost penalty per extra share bought (default: 0.05)
        theta_queue: Queue-risk penalty parameter (default: 0.0005)
        
    Returns:
        float: Total cost of execution
    """
    order_size = 5000  # Total shares to buy
    orders_filled = 0  # Running count of shares filled
    time_idx = 0      # Current position in the market data
    total_cost = 0    # Accumulator for total execution cost
    
    while orders_filled < order_size and time_idx < len(df):
        # Get current venue list
        venue_list = df['venue'].iloc[time_idx]
        
        # Calculate remaining shares to fill
        remaining_size = order_size - orders_filled
        
        # Get optimal split using Cont-Kukanov allocation
        split, cost = allocate(
            order_size=remaining_size,
            venues=venue_list,
            lambda_over=lam_over,
            lambda_under=lam_under,
            theta_queue=theta_queue
        )

        if split is None:
            time_idx += 1
            continue
        
        # Execute the split and update totals
        executed = sum(min(s, v.ask_sz) for s, v in zip(split, venue_list))
        orders_filled += executed
        total_cost += cost
        
        # Move to next tick
        time_idx += 1
        
    return total_cost

def optimize_contkukanov(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Optimize the Cont-Kukanov model parameters for the given market data.
    
    Args:
        df: DataFrame with market data indexed by timestamp
        
    """
    grid = product(np.logspace(-5, 2, 30),
                   np.logspace(-5, 2, 30),
                   np.logspace(-5, 2, 30))
    
    best_cost = float('inf')
    best_params = None

    for lam_under, lam_over, theta_queue in grid:
        cost = backtest_contkukanov(df, lam_under, lam_over, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_params = (lam_under, lam_over, theta_queue)

    return best_cost, best_params
    
def optimize_contkukanov_parallel(df: pd.DataFrame) -> Tuple[float, Tuple[float, float, float]]:
    """
    Optimize the Cont-Kukanov model parameters for the given market data using parallel processing.
    
    Args:
        df: DataFrame with market data indexed by timestamp
        
    Returns:
        Tuple[float, Tuple[float, float, float]]: Best cost and corresponding parameters 
        (lambda_under, lambda_over, theta_queue)
    """
    # Generate parameter grid
    param_grid = list(product(
        np.logspace(-5, 2, 30),  # lambda_under
        np.logspace(-5, 2, 30),  # lambda_over
        np.logspace(-5, 2, 30)   # theta_queue
    ))
    
    # Get number of available CPU cores
    num_cores = multiprocessing.cpu_count()
    max_workers = min(num_cores * 2, len(param_grid))  # Use 2 threads per core
    print(f"Optimizing using {max_workers} threads across {num_cores} CPU cores")
    
    def evaluate_params(params: Tuple[float, float, float]) -> Tuple[float, Tuple[float, float, float]]:
        """Helper function to evaluate a single parameter set"""
        lam_under, lam_over, theta_queue = params
        cost = backtest_contkukanov(df, lam_under, lam_over, theta_queue)
        return cost, params
    
    best_cost = float('inf')
    best_params = None
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all parameter combinations for evaluation
        future_to_params = {
            executor.submit(evaluate_params, params): params 
            for params in param_grid
        }
        
        # Process results as they complete
        completed = 0
        total_tasks = len(param_grid)
        
        for future in concurrent.futures.as_completed(future_to_params):
            completed += 1
            try:
                cost, params = future.result()
                if cost < best_cost:
                    best_cost = cost
                    best_params = params
                    print(f"New best cost {best_cost:.2f} found with parameters {best_params}")
            except Exception as e:
                print(f"Parameter evaluation failed: {e}")
    
    if best_params is None:
        raise RuntimeError("Optimization failed to find valid parameters")
        
    return best_cost, best_params

if __name__ == '__main__':
    df = parse('l1_day_changed_publisher_id.csv')
    print(f'Naive Best Ask: {backtest_take_best_ask(df)}')
    print(f'TWAP: {backtest_twap(df)}')
    print(f'VWAP: {backtest_vwap(df)}')
    print(f'Default Cont-Kukanov: {backtest_contkukanov(df)}')


    print("\nOptimizing Cont-Kukanov parameters...")
    # best_cost, (best_lam_under, best_lam_over, best_theta_queue) = optimize_contkukanov_parallel(df)
    best_cost, (best_lam_under, best_lam_over, best_theta_queue) = optimize_contkukanov(df)
    print("\nOptimization Results:")
    print(f'Best Cost: {best_cost:.2f}')
    print(f'Best Parameters:')
    print(f'  lambda_under: {best_lam_under:.6f}')
    print(f'  lambda_over:  {best_lam_over:.6f}')
    print(f'  theta_queue:  {best_theta_queue:.6f}')
