# Can a neural network hedge a barrier reverse convertible better than Black-Scholes?

Semester project, Advanced Data Analytics, HEC Lausanne (Prof. D. Karyampas), autumn 2026.
Author: Saad Chraibi.

---

## 1. The question in one paragraph

Barrier reverse convertibles (BRCs) are among the most sold structured products in
Switzerland. The investor earns a fixed coupon but **sells a down-and-in put** to the
issuing bank. The bank therefore holds that put and must hedge it by trading the
underlying index. The put is hard to hedge: its delta jumps when the index nears the
barrier, so hedging errors pile up in crashes, exactly when they hurt most. **This
project tests, on real market paths from 2000 to 2026, whether hedges learned by
machine learning lose less in the tail than the textbook Black-Scholes delta hedge.**

## 2. The product (fixed before any result is seen)

| Feature | Base case | Robustness |
|---|---|---|
| Underlying | S&P 500 | SMI |
| Maturity | 1 year (252 trading days) | – |
| Strike | 100 % of the index on issue date | – |
| Barrier | 70 % of issue level | 60 %, 80 % |
| Barrier monitoring | daily, on closing prices | – |
| Coupon | fair coupon from the pricing model on issue date | – |
| Issue dates | every trading day (overlapping products) | monthly (non-overlapping check) |

Decomposition: **investor = zero-coupon bond + coupon − down-and-in put**.
The bank is long the down-and-in put and hedges it with the index.

## 3. Hedging strategies compared

| # | Strategy | Role |
|---|---|---|
| 1 | No hedge | Floor |
| 2 | Black-Scholes delta, rebalanced daily | Industry benchmark |
| 3 | Black-Scholes delta with a no-trade band | Benchmark that saves costs |
| 4 | Feed-forward neural network | ML hedge (weeks 7-8) |
| 5 | LSTM using recent price history | ML hedge (week 8) |
| 6 | Reinforcement-learning hedge | Optional extension (week 13) |

The neural networks output a hedge ratio each day and are trained to **minimise the
Expected Shortfall (97.5 %) of the final hedging P&L**, including transaction costs.
Market regimes (calm, stressed) are found with a **Gaussian mixture model** (week 12)
and used both as a network input and to split the results.

## 4. Evaluation (fixed in advance)

- Out-of-sample only: train on early years, test on later years, walking forward.
  No product in the test set may overlap the training period.
- Metrics on hedging P&L: **Expected Shortfall 97.5 %**, mean, standard deviation, turnover.
- Results reported separately for calm periods and for 2008, 2020 and 2022.
- **Hypothesis:** learned hedges reduce tail losses near the barrier.
  A null result is reported as such: it would mean delta hedging is hard to beat on real paths.

## 5. Data

| Data | Source | Used for |
|---|---|---|
| S&P 500 price and total-return index | Yahoo Finance | Underlying, dividend yield |
| SMI | Yahoo Finance | Robustness underlying |
| VIX, VIX3M | FRED | Implied volatility (pricing, network input) |
| SKEW index | CBOE | Price of crash protection (network input) |
| US 3-month and 1-year rates | FRED | Discounting |
| Swiss 3-month rate (monthly) | FRED / OECD | Discounting, SMI |
| Implied volatility by moneyness and maturity | CEDIF (Bloomberg or OptionMetrics) | The volatility smile, see `data/cedif/README.md` |

## 6. Known limits (to state in the paper)

- **Volatility smile.** Without CEDIF data, the put is priced with one volatility (VIX),
  which underprices it, since out-of-the-money puts trade at higher implied volatility.
  Both hedges face the same pricing model, so the comparison stays fair; P&L is measured on real paths.
- **Maturity mismatch.** The VIX is a 30-day volatility; the product lasts one year.
  The 12-month implied volatility from CEDIF fixes this.
- **Overlapping products** are not independent: hence walk-forward splits and a monthly check.
- **Few crises.** Only a handful of real stress episodes, so tail results carry wide uncertainty.
- **Swiss rates** are monthly on FRED; replace with daily SARON if available.

## 7. Reproduce the data

```bash
pip install -r requirements.txt
python src/download_data.py   # raw files    -> data/raw/
python src/clean_data.py      # clean panels -> data/processed/  (+ data_report.md)
python src/make_figures.py    # figures      -> figures/
```

Outputs: `panel_spx.csv` and `panel_smi.csv`, one row per trading day with price, log
return, realised volatility, VIX, VIX3M, SKEW, rates, dividend yield and the volatility
risk premium.

## 8. Running time and resources (requirement 4.24)

Everything runs in the course's Nuvolos workspace; no other resource is needed.
Data download and cleaning: about 2 minutes. Model training: to be measured and stated
here once the models exist (target: under 30 minutes on CPU for the full run).

## 9. Data licences (requirement 4.15)

| Data | Committed to this repository? |
|---|---|
| FRED series (rates) | To be checked per series: some FRED series are public domain, others belong to their publisher |
| VIX, VIX3M, SKEW (CBOE, via FRED or CBOE) | Check CBOE terms before committing; otherwise obtained by `download_data.py` |
| Yahoo Finance prices | Yahoo's terms restrict redistribution: obtained by `download_data.py` |
| CEDIF (Bloomberg / OptionMetrics) | No: licensed data, not redistributable in a public repository. Instructions in `data/cedif/README.md` |

Where data cannot be committed, the code that obtains it and this statement of its licence
are included instead, as requirement 4.15 allows. The core results must not depend on
CEDIF data, so that anyone can reproduce them from free sources.

## 10. Repository layout (requirements 4.7-4.9)

```
src/            shared Python modules (data download and cleaning; later: pricing,
                hedging, ES), imported wherever needed, never copied
notebooks/      one notebook per stage, each runs top to bottom in a fresh kernel:
                01_data_check, 02_pricing_and_delta_hedge, 03_ml_hedges, 04_results
data/           raw/ and processed/ (where licences allow), cedif/ instructions
figures/        every figure in the paper, produced by code in this repository
paper/          the SIAM paper (PDF) and its sources
```

## 11. What I code myself (the interview-critical parts)

- Closed-form price and delta of a down-and-in put (Reiner-Rubinstein, in Hull).
- The hedging P&L loop with transaction costs.
- Expected Shortfall and the walk-forward split.

## 12. AI use

Data scripts drafted with Claude (Anthropic), then reviewed, tested and run by the author.
To be listed in the paper's appendix, as the course requires.
