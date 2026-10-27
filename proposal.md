Subject: ADA 2026 proposal — Chraibi Saad

**Can a neural network hedge a barrier reverse convertible better than Black-Scholes?**

**Question.** Barrier reverse convertibles are among the most popular structured products in Switzerland. The issuing bank holds a down-and-in put whose delta jumps near the barrier, so hedging losses concentrate in crashes. I test whether hedges learned by machine learning reduce these tail losses compared with Black-Scholes delta hedging.

**Data.** Daily S&P 500 (price and total return) and SMI, 1990–2026, about 9,000 trading days each (Yahoo Finance); VIX, VIX3M and SKEW implied-volatility indices (FRED, CBOE); US and Swiss short rates (FRED). Downloaded and cleaned. If licensing allows, implied volatility by moneyness from CEDIF.

**Method.** On each trading day I issue a one-year product (barrier 70%), price its put in closed form, and hedge it daily on the realised path with transaction costs. I compare Black-Scholes delta, delta with a no-trade band, and feed-forward and LSTM networks trained to minimise the Expected Shortfall of hedging P&L; a Gaussian mixture model identifies regimes. Evaluation is walk-forward and out-of-sample, overall and in 2008, 2020 and 2022. If delta hedging is not beaten, that is the finding.
