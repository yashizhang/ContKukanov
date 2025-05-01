import os 
from typing import List, Tuple
from dataclasses import dataclass
import pandas as pd 
import numpy as np 
import json


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

def allocate(order_size: int, 
             venues: List[Venue], 
             lam_over: float, 
             lam_under: float, 
             theta_queue: float) -> Tuple[List[int], float]:
    step = 100
    splits = [[]]
    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v].ask_sz)
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

def compute_cost(split: List[int], 
                 venues: List[Venue], 
                 order_size: int, 
                 lam_over: float, 
                 lam_under: float, 
                 theta_queue: float) -> float:
    executed = 0
    cash_spent = 0.0
    for i in range(len(venues)):
        exe = min(split[i], venues[i].ask_sz)
        executed += exe
        cash_spent += exe * (venues[i].ask + venues[i].fee)
        maker_rebate = max(split[i] - exe, 0) * venues[i].rebate
        cash_spent -= maker_rebate
    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta_queue * (underfill + overfill)
    cost_pen = lam_under * underfill + lam_over * overfill
    return cash_spent + risk_pen + cost_pen

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
        best_ask_idx = venue_list.index(best_ask)

        # Create order split targeting only the best ask venue
        split = [0 for _ in range(len(venue_list))]
        # Take as many shares as possible from best ask, up to what we still need
        split[best_ask_idx] = min(order_size - orders_filled, best_ask.ask_sz)

        # Calculate cost for this execution
        cost = compute_cost(split, venue_list, order_size, 0, 0, 0)
        # Update running totals
        orders_filled += split[best_ask_idx]
        total_cost += cost

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
                
                # Create split order targeting venues closest to TWAP
                split = [0] * len(venues)
                for i, venue in enumerate(sorted_venues):
                    if orders_filled >= order_size:
                        break
                        
                    # Calculate how many shares to buy from this venue
                    shares_needed = min(
                        order_size - orders_filled,  # Shares still needed
                        venue.ask_sz,  # Available size at venue
                        order_size // len(bucket_prices)  # Roughly equal distribution across bucket
                    )
                    
                    if shares_needed > 0:
                        venue_idx = venues.index(venue)
                        split[venue_idx] = shares_needed
                        
                # Execute the split and compute cost
                cost = compute_cost(split, venues, order_size, 0, 0, 0)
                orders_filled += sum(split)
                total_cost += cost
            
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
                
                # Create split order targeting venues closest to VWAP
                split = [0] * len(venues)
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
                        venue_idx = venues.index(venue)
                        split[venue_idx] = shares_needed
                        
                # Execute the split and compute cost
                cost = compute_cost(split, venues, order_size, 0, 0, 0)
                orders_filled += sum(split)
                total_cost += cost
            
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

def backtest_contkukanov(df: pd.DataFrame) -> Tuple[float, dict]:
    """
    Implement the Cont-Kukanov strategy with parameter optimization.
    First performs grid search to find optimal parameters, then executes the strategy.
    
    Args:
        df: DataFrame with market data indexed by timestamp
        
    Returns:
        Tuple[float, dict]: Total execution cost and dictionary of optimal parameters
    """
    order_size = 5000
    
    # First phase: Grid search for optimal parameters
    def grid_search() -> Tuple[float, float, float]:
        best_params = None
        best_total_cost = float('inf')
        
        # Create grid of parameters
        param_values = np.linspace(0, 1, 100)  # 10 subdivisions in [0,1]
        
        # Try first 100 timestamps for parameter optimization
        sample_size = min(100, len(df))
        sample_df = df.iloc[:sample_size]
        
        for lam_over in param_values:
            for lam_under in param_values:
                for theta_queue in param_values:
                    total_cost = 0
                    orders_filled = 0
                    
                    # Test these parameters on sample data
                    for time_idx in range(sample_size):
                        if orders_filled >= order_size:
                            break
                            
                        venue_list = sample_df['venue'].iloc[time_idx]
                        
                        # Get optimal split using Cont-Kukanov allocator
                        remaining = order_size - orders_filled
                        split, cost = allocate(
                            remaining, venue_list, 
                            lam_over, lam_under, theta_queue
                        )
                        
                        orders_filled += sum(split)
                        total_cost += cost
                        
                    # Update best parameters if current ones are better
                    if total_cost < best_total_cost and orders_filled == order_size:
                        best_total_cost = total_cost
                        best_params = (lam_over, lam_under, theta_queue)
        
        return best_params
    
    # Run grid search to find optimal parameters
    print("Running grid search for optimal parameters...")
    optimal_params = grid_search()
    
    if optimal_params is None:
        raise ValueError("Could not find valid parameters that complete the order")
        
    lam_over, lam_under, theta_queue = optimal_params
    print(f"Optimal parameters found: λ_over={lam_over:.4f}, λ_under={lam_under:.4f}, θ_queue={theta_queue:.4f}")
    
    # Second phase: Execute strategy with optimal parameters
    orders_filled = 0
    time_idx = 0
    total_cost = 0
    
    while orders_filled < order_size and time_idx < len(df):
        venue_list = df['venue'].iloc[time_idx]
        
        # Get optimal split using Cont-Kukanov allocator with optimized parameters
        remaining = order_size - orders_filled
        split, cost = allocate(
            remaining, venue_list,
            lam_over, lam_under, theta_queue
        )
        
        orders_filled += sum(split)
        total_cost += cost
        time_idx += 1
    
    # Package optimal parameters in dictionary
    optimal_params_dict = {
        'lambda_over': lam_over,
        'lambda_under': lam_under,
        'theta_queue': theta_queue
    }
    
    return total_cost, optimal_params_dict

if __name__ == '__main__':
    df = parse('l1_day.csv')
    print(f'Naive Best Ask: {backtest_take_best_ask(df)}')
    print(f'TWAP: {backtest_twap(df)}')
    print(f'VWAP: {backtest_vwap(df)}')
    cost, params = backtest_contkukanov(df)
    print(f'Cont-Kukanov cost: {cost}')
    print(f'Optimal parameters: {params}')
