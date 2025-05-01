# Implementation of the [Cont-Kukanov Optimal Order Execution Model](https://arxiv.org/pdf/1210.1625) #


## Setup ##
```bash
git clone https://github.com/yashizhang/ContKukanov.git
conda create -p ./conda_env python=3.12
conda activate ./conda_env
pip install numpy pandas
```

## Assumptions ##
1. Since the dataset we are using contains tick-level data for high-volume 
U.S. equities, and do not mention rebates and fees for taking/providing 
liquidity, we derive the following values based on March 2025 guidance by 
the NASDAQ and NYSE:

    | Rebates | Fees |
    |---------|------|
    | $0.0030 | $0.0000 |

    Sources: 
    [NASDAQ Trader News](https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2025-13)
    [NYSE National Inc.](https://www.nyse.com/publicdocs/nyse/regulation/nyse/NYSE_National_Schedule_of_Fees.pdf)

    Note: 
    Fees are free until you exceed 4 million shares removes per month in NASDAQ. NYSE has basically the same rebate rate ($0.0029), and no charge for removing liquidity if the order executes at a price better than the contra-side NBBO and ($0.0016) otherwise. To keep things simple and efficient, we take fees to be $0. 

    I also had the option of using the values in the Cont & Kukanov paper, but the values they use - while current at the time - are now ~13 years outdated. By using more accurate fees and rebates, the model should perform slightly better as the optimized values of lambda_under, lambda_over, and theta_queue will be with respect to the true cost function. 

2. For simplicity and due to personal time constraints, I will not be modeling our own market impact. Ideally, every time we execute a trade of order size `sz` at price `ask`, we should update the following ticks' orderbooks size at price level `ask` to be `original_size - sz` (i.e. actually model the liquidity being taken away by our market order). 

3. I added a condition in the allocate function: if there are no good splits to check through (e.g. the best ask size is too small), we return None, None. This way, the backtest function knows to skip the current timestamp and go to the next timestamp. 

## Code Structure ##
1. Data Classes and Parsing
* Venue class: Represents a trading venue with ask price, size, fee, and rebate 
* parse(): Preprocesses market data from CSV into DataFrame with venue information

2. Core Trading Functions
* compute_cost(): Calculates execution cost given a split across venues
* allocate(): Optimizes order allocation across venues to minimize cost

3. Trading Strategies

    All take DataFrame input and return cost:
* backtest_take_best_ask(): Simple strategy taking best available ask price
* backtest_twap(): Time-Weighted Average Price implementation
* backtest_vwap(): Volume-Weighted Average Price implementation
* backtest_contkukanov(): Cont-Kukanov optimal execution mod

4. Optimization Functions
* optimize_contkukanov(): Single-threaded parameter optimization
* optimize_contkukanov_parallel(): Multi-threaded version using ThreadPoolExecutor

5. Main Execution
* Parses command line arguments
* Runs backtests with default parameters
* Performs grid search to optimize risk parameters (λ_under, λ_over, θ_queue)

The code implements and compares different trading execution strategies with a focus on the Cont-Kukanov model, allowing for both default and optimized risk parameters.

## Bugs ##
* Either the dataset given is not representative enough or the pseudo-code is not fully correct, leading to trivial optimizations with respect to the risk parameters. 
* The data given only contains order book data from a single venue, which may affect correctness and may show lack of performance increase when there should be. 