# Implementation of the Cont-Kukanov Optimal Order Execution Model #


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

3. I also did not use a validation test to validate if the optimized lambda_under, lambda_over, and theta_queue functions were being overfit. This boils down to an time-invariant market regime assumption (?).