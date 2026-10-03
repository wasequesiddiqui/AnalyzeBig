# Comprehensive Forensic Stock Analysis Framework

## 1. Purpose of Forensic Stock Analysis

A forensic analysis framework should help answer:

- Are revenues genuine and sustainable?
- Are reported profits converting into cash?
- Are assets real and productive?
- Are liabilities understated?
- Is management using accounting choices to improve reported results?
- Is the company financially fragile despite apparently strong earnings?
- Are related parties, promoters, auditors, subsidiaries, or associates creating hidden risks?
- Does the accounting and financial profile resemble known manipulation or failure patterns?

---

# 2. Core Established Forensic and Financial Quality Models

The major models include:

1. Beneish M-Score
2. Dechow F-Score
3. Piotroski F-Score
4. Altman Z-Score
5. Altman Z′-Score
6. Altman Z″-Score
7. Sloan Accrual Model
8. DuPont Analysis
9. Jones Model
10. Modified Jones Model
11. Kasznik Model
12. Beneish 5-variable M-Score

---

# 3. Beneish M-Score

## 3.1 Original 8-Variable Beneish Model

### DSRI — Days' Sales in Receivables Index

\[
DSRI =
\frac{AR_t / Revenue_t}
{AR_{t-1} / Revenue_{t-1}}
\]

### GMI — Gross Margin Index

\[
GMI =
\frac{GrossMargin_{t-1}}
{GrossMargin_t}
\]

where:

\[
GrossMargin =
\frac{Revenue-COGS}{Revenue}
\]

### AQI — Asset Quality Index

\[
AQI =
\frac{
1-\frac{CA_t+PPE_t+Securities_t}{TA_t}
}{
1-\frac{CA_{t-1}+PPE_{t-1}+Securities_{t-1}}{TA_{t-1}}
}
\]

### SGI — Sales Growth Index

\[
SGI =
\frac{Revenue_t}{Revenue_{t-1}}
\]

### DEPI — Depreciation Index

\[
DEPI =
\frac{
Dep_{t-1}/(PPE_{t-1}+Dep_{t-1})
}{
Dep_t/(PPE_t+Dep_t)
}
\]

### SGAI — Sales, General & Administrative Expense Index

\[
SGAI =
\frac{SG\&A_t/Revenue_t}
{SG\&A_{t-1}/Revenue_{t-1}}
\]

### LVGI — Leverage Index

\[
LVGI =
\frac{(CL_t+LTD_t)/TA_t}
{(CL_{t-1}+LTD_{t-1})/TA_{t-1}}
\]

### TATA — Total Accruals to Total Assets

\[
TATA =
\frac{IncomeFromContinuingOperations_t-CFO_t}
{TA_t}
\]

### Beneish M-Score

\[
M =
-4.84
+0.920DSRI
+0.528GMI
+0.404AQI
+0.892SGI
+0.115DEPI
-0.172SGAI
+4.679TATA
-0.327LVGI
\]

### Benchmark

| M-Score | Interpretation |
|---|---|
| < -1.78 | Lower manipulation likelihood |
| > -1.78 | Elevated manipulation likelihood |

**Important:** This is a probabilistic screening model, not proof of fraud.

## 3.2 Beneish Component Screening Benchmarks

| Component | Generally Healthy | Warning |
|---|---:|---:|
| DSRI | ~1.00 | >1.20–1.25 |
| GMI | ≤1.00 | >1.10 |
| AQI | ≤1.00 | >1.10–1.20 |
| SGI | ~1.00–1.20 | >1.50 |
| DEPI | ~1.00 | >1.10 |
| SGAI | ~1.00 | >1.10–1.20 |
| LVGI | ≤1.00 | >1.10 |
| TATA | ~0 or negative | >5% |
| M-Score | < -1.78 | > -1.78 |

These component levels are **practical screening thresholds**, not official individual Beneish cutoffs.

---

# 4. Dechow F-Score

## 4.1 Dechow Model 1

The financial-statement-based model can be represented as:

\[
logit =
-7.893
+0.790RSSTACC
+2.518\Delta REC
+1.191\Delta INV
+1.979SOFTASSETS
+0.171\Delta CASHSALES
-0.932\Delta ROA
+1.029ISSUE
\]

Probability:

\[
P(FSF)=
\frac{e^{logit}}
{1+e^{logit}}
\]

F-Score:

\[
F =
\frac{P(FSF)}
{0.0037}
\]

## 4.2 Variables

### RSSTACC

Measures changes in accrual components scaled by average total assets, following the Dechow et al. methodology.

### ΔREC

\[
\Delta REC =
\frac{AR_t-AR_{t-1}}
{AverageTA}
\]

### ΔINV

\[
\Delta INV =
\frac{Inventory_t-Inventory_{t-1}}
{AverageTA}
\]

### SOFTASSETS

\[
SOFTASSETS =
\frac{TA-NetPPE-Cash}{TA}
\]

### ΔCASHSALES

\[
\Delta CASHSALES =
\left[
\frac{Sales_t-\Delta AR_t}
{Sales_{t-1}-\Delta AR_{t-1}}
\right]-1
\]

### ΔROA

\[
\Delta ROA=ROA_t-ROA_{t-1}
\]

### ISSUE

Binary variable:

- 1 = long-term debt or common stock issued
- 0 = otherwise

## 4.3 Benchmark

A useful conventional interpretation is:

| F-Score | Interpretation |
|---:|---|
| <1.00 | Below-normal risk |
| ≥1.00 | Above-normal risk |
| ≥1.85 | Substantial risk |
| ≥2.45 | High risk |

These should be treated as conventional/published interpretations rather than universal fraud thresholds.

**Important:** Dechow developed multiple models. Models 2 and 3 incorporate additional non-financial, market, or off-balance-sheet information.

---

# 5. Piotroski F-Score

Piotroski uses nine binary signals.

## 5.1 Profitability

| Signal | Condition | Score |
|---|---|---:|
| Positive ROA | ROA > 0 | 1 |
| Positive CFO | CFO > 0 | 1 |
| ΔROA | Current ROA > Previous ROA | 1 |
| CFO > NI | CFO exceeds Net Income | 1 |

## 5.2 Leverage, Liquidity and Source of Funds

| Signal | Condition | Score |
|---|---|---:|
| Lower leverage | Leverage decreases | 1 |
| Higher current ratio | Current ratio increases | 1 |
| No equity dilution | No new equity issued | 1 |

## 5.3 Operating Efficiency

| Signal | Condition | Score |
|---|---|---:|
| Higher gross margin | Gross margin increases | 1 |
| Higher asset turnover | Asset turnover increases | 1 |

### Total

\[
PiotroskiF =
\sum_{i=1}^{9}Signal_i
\]

Maximum = 9.

### Benchmark

| Score | Interpretation |
|---:|---|
| 8–9 | Strong |
| 5–7 | Mixed |
| 3–4 | Weak |
| 0–2 | Very weak / distressed |

**Important:** Piotroski was designed primarily for value-stock selection and financial strength, not specifically for fraud detection.

---

# 6. Altman Z-Score

## 6.1 Original Public Manufacturing Model

\[
Z =
1.2\frac{WC}{TA}
+1.4\frac{RE}{TA}
+3.3\frac{EBIT}{TA}
+0.6\frac{MVE}{TL}
+1.0\frac{Sales}{TA}
\]

Where:

- WC = Working Capital
- TA = Total Assets
- RE = Retained Earnings
- EBIT = Earnings Before Interest and Taxes
- MVE = Market Value of Equity
- TL = Total Liabilities

### Benchmark

| Z-Score | Zone |
|---:|---|
| >2.99 | Safe |
| 1.81–2.99 | Grey |
| <1.81 | Distress |

Do not blindly apply the original model to banks, insurers, or every non-manufacturing company.

---

# 7. Altman Z′ and Z″

Different Altman variants use different coefficients and thresholds.

## Altman Z′

A commonly used private-company formulation uses:

\[
Z' =
0.717A+
0.847B+
3.107C+
0.420D+
0.998E
\]

where the variables are scaled balance-sheet/profitability measures appropriate to the model.

A commonly cited screening interpretation is:

| Z′ | Interpretation |
|---:|---|
| >2.90 | Safe |
| 1.23–2.90 | Grey |
| <1.23 | Distress |

**Always confirm the exact variant before implementation.**

## Altman Z″ — Non-Manufacturing Variant

A commonly used non-manufacturing formulation is:

\[
Z'' =
6.56A+
3.26B+
6.72C+
1.05D
\]

with:

- A = WC / TA
- B = RE / TA
- C = EBIT / TA
- D = Book Equity / Total Liabilities

### Benchmark

| Z″ | Interpretation |
|---:|---|
| >2.60 | Safe |
| 1.10–2.60 | Grey |
| <1.10 | Distress |

---

# 8. Sloan Accrual Model

A common forensic implementation is:

\[
SloanAccrual =
\frac{NI-(CFO-Capex)}
{AverageTA}
\]

Other literature uses variants such as:

\[
AccrualRatio =
\frac{NI-CFO}{AverageTA}
\]

Therefore, the implementation must explicitly define which formulation is used.

### Practical Screening

| Accrual Ratio | Interpretation |
|---:|---|
| <5% | Generally healthy |
| 5–10% | Caution |
| >10% | High accrual intensity |

These are screening conventions rather than universal academic thresholds.

---

# 9. Jones Model

\[
\frac{TA_{i,t}}{A_{i,t-1}}
=
\alpha_1\frac{1}{A_{i,t-1}}
+\beta_1\frac{\Delta REV_{i,t}}{A_{i,t-1}}
+\beta_2\frac{PPE_{i,t}}{A_{i,t-1}}
+\epsilon_{i,t}
\]

The residual represents abnormal/discretionary accruals.

### Interpretation

| Abnormal Accrual | Interpretation |
|---|---|
| Around 0 | Normal |
| Positive and persistent | Possible income-increasing earnings management |
| Large positive values | Stronger warning |
| Large negative values | Possible income-decreasing management |

There is **no universal fixed threshold**. Compare against industry-year distributions.

---

# 10. Modified Jones Model

\[
\frac{TA}{A_{prev}}
=
\alpha_1\frac{1}{A_{prev}}
+\beta_1
\frac{\Delta REV-\Delta REC}{A_{prev}}
+\beta_2\frac{PPE}{A_{prev}}
+\epsilon
\]

The residual is the estimated abnormal accrual.

### Benchmark

- Around 0 = normal
- Large positive abnormal accrual = possible income-increasing earnings management
- Persistent abnormal accrual = stronger warning
- Compare against industry and peer distributions

---

# 11. Kasznik Model

\[
\frac{TA}{A_{prev}}
=
\alpha_1\frac{1}{A_{prev}}
+\beta_1\frac{\Delta REV}{A_{prev}}
+\beta_2\frac{PPE}{A_{prev}}
+\beta_3\frac{\Delta CFO}{A_{prev}}
+\epsilon
\]

The residual represents abnormal accruals.

Again, there is no universal fixed cutoff.

---

# 12. DuPont Analysis

\[
ROE =
NetMargin
\times
AssetTurnover
\times
FinancialLeverage
\]

Expanded:

\[
ROE =
\frac{NI}{Revenue}
\times
\frac{Revenue}{Assets}
\times
\frac{Assets}{Equity}
\]

### Forensic Interpretation

A high ROE can arise from:

- High margins
- High asset efficiency
- High leverage

Therefore, a high ROE is not automatically high quality.

---

# 13. Revenue Quality Ratios

| Metric | Formula |
|---|---|
| Revenue Growth | Revenue_t / Revenue_t-1 − 1 |
| Receivables Growth | AR_t / AR_t-1 − 1 |
| AR / Revenue | AR / Revenue |
| ΔAR / ΔRevenue | ΔAR / ΔRevenue |
| DSO | Avg AR / Revenue × 365 |
| DSO Change | DSO_t − DSO_t-1 |
| AR / Assets | AR / Assets |
| Contract Assets / Revenue | Contract Assets / Revenue |
| Unbilled Revenue / Revenue | Unbilled Revenue / Revenue |
| Deferred Revenue Growth | Deferred Revenue Growth |
| Cash Collections / Revenue | Cash Collections / Revenue |
| CFO / Revenue | CFO / Revenue |
| CFO / EBITDA | CFO / EBITDA |
| CFO / EBIT | CFO / EBIT |
| Revenue Concentration | Largest customer/segment revenue share |
| Top-5 Customer Concentration | Top 5 customers / Revenue |
| Revenue / Employee | Revenue / Employees |
| Revenue / Assets | Revenue / Average Assets |

### Revenue Growth Mismatch

\[
RRG =
\frac{\Delta AR/AR_{prev}}
{\Delta Revenue/Revenue_{prev}}
\]

### Screening

| Test | Healthy | Warning |
|---|---|---|
| DSO | Stable / peer-normal | >10–15% YoY increase |
| ΔAR / ΔRevenue | <1 | >1–1.5 |
| AR growth / Revenue growth | <1 | >1.5 |
| Unbilled/Contract Assets | Stable | Rapid increase |
| CFO / Revenue | Stable | Persistent decline |

---

# 14. Earnings Quality

| Metric | Formula |
|---|---|
| Cash Earnings Ratio | CFO / NI |
| Accrual Ratio | (NI − CFO) / Avg TA |
| Accrual Gap | NI − CFO |
| Accrual / Revenue | (NI − CFO) / Revenue |
| Accrual / Assets | (NI − CFO) / Avg Assets |
| FCF Conversion | FCF / NI |
| EBITDA-to-CFO | CFO / EBITDA |
| EBIT-to-CFO | CFO / EBIT |
| CFO-to-PAT | CFO / PAT |
| FCF | CFO − Capex |
| CFO Growth vs PAT Growth | Compare growth rates |

### Practical Benchmarks

| Metric | Good | Caution | Severe Warning |
|---|---:|---:|---:|
| CFO / NI | >1.0 | 0.7–1.0 | <0.7 |
| FCF / NI | >0.8 | 0–0.8 | <0 |
| CFO / EBITDA | >0.8 | 0.6–0.8 | <0.6 |
| CFO / EBIT | >1.0 | 0.7–1.0 | <0.7 |
| Accrual / Assets | <3% | 3–5% | >5% |
| Accrual / Revenue | <3% | 3–5% | >5% |

---

# 15. Liquidity Ratios

| Metric | Formula |
|---|---|
| Current Ratio | CA / CL |
| Quick Ratio | (Cash + Securities + AR) / CL |
| Cash Ratio | (Cash + Securities) / CL |
| Working Capital / Assets | WC / Assets |
| CFO / Current Liabilities | CFO / CL |
| CFO / Total Debt | CFO / Debt |
| Interest Coverage | EBIT / Interest |
| Cash Interest Coverage | CFO / Interest Paid |
| DSCR | CFO / (Interest + Principal Repayment) |

### Benchmarks

| Ratio | Good | Caution | Severe |
|---|---:|---:|---:|
| Current Ratio | >1.5 | 1–1.5 | <1 |
| Quick Ratio | >1.0 | 0.7–1.0 | <0.7 |
| Cash Ratio | >0.5 | 0.2–0.5 | <0.2 |
| Interest Coverage | >5 | 2–5 | <2 |
| CFO / Interest | >5 | 2–5 | <2 |

Industry context is essential.

---

# 16. Debt and Leverage

| Metric | Formula |
|---|---|
| Debt / Equity | Debt / Equity |
| Liabilities / Equity | Liabilities / Equity |
| Debt / Assets | Debt / Assets |
| Net Debt / EBITDA | Net Debt / EBITDA |
| Debt / CFO | Debt / CFO |
| Long-Term Debt / Capital | LTD / (LTD + Equity) |
| Financial Leverage | Avg Assets / Avg Equity |
| Debt Growth | Debt Growth % |
| Debt Growth vs EBITDA Growth | Compare growth rates |
| Debt Maturity Concentration | Debt due within period / Total Debt |
| Short-Term Debt / Total Debt | STD / Total Debt |

### Benchmarks

| Ratio | Good | Caution | Severe |
|---|---:|---:|---:|
| Debt / Equity | <1 | 1–2 | >2 |
| Debt / Assets | <40% | 40–60% | >60% |
| Net Debt / EBITDA | <2 | 2–3 | >3 |
| Debt / CFO | <3 | 3–5 | >5 |
| Interest Coverage | >5 | 2–5 | <2 |

---

# 17. Working Capital

| Metric | Formula |
|---|---|
| DSO | Avg AR / Revenue × 365 |
| DIO | Avg Inventory / COGS × 365 |
| DPO | Avg AP / COGS × 365 |
| Cash Conversion Cycle | DSO + DIO − DPO |
| ΔCCC | CCC_t − CCC_t-1 |
| Inventory Growth − Sales Growth | %ΔInventory − %ΔRevenue |
| AP Growth − COGS Growth | %ΔAP − %ΔCOGS |
| Working Capital / Revenue | WC / Revenue |
| ΔAR − ΔRevenue | Change in AR minus change in Revenue |
| ΔInventory − ΔCOGS | Change in Inventory minus change in COGS |
| ΔAP − ΔCOGS | Change in AP minus change in COGS |
| ΔWorking Capital / Revenue | ΔWC / Revenue |

### Screening

- DSO, DIO and CCC should generally be stable and peer-normal.
- >20% above peer levels is a warning signal.
- Inventory growth > revenue growth by >10 percentage points is a warning.
- AP growth > COGS growth by >15 percentage points can indicate unusual working-capital support.

---

# 18. Inventory Forensics

| Metric | Formula |
|---|---|
| Inventory Turnover | COGS / Avg Inventory |
| Inventory / Revenue | Inventory / Revenue |
| Inventory Growth | YoY inventory growth |
| Inventory Growth Premium | %ΔInventory − %ΔRevenue |
| DIO | Avg Inventory / COGS × 365 |
| Provision / Inventory | Inventory Provision / Inventory |
| Write-off Ratio | Inventory Write-offs / Inventory |

Warning signs include:

- Inventory growing much faster than sales.
- DIO rising persistently.
- Falling turnover.
- Low or declining inventory provisions despite rising inventory risk.
- Repeated inventory write-offs.

---

# 19. Fixed Assets and Capex

| Metric | Formula |
|---|---|
| Capex / Revenue | Capex / Revenue |
| Capex / Depreciation | Capex / Depreciation |
| PPE Growth / Revenue Growth | %ΔPPE / %ΔRevenue |
| Depreciation / Gross PPE | Depreciation / Gross PPE |
| Average Asset Age | Accumulated Depreciation / Depreciation |
| Capex / CFO | Capex / CFO |
| Free Cash Flow | CFO − Capex |

### Screening

| Test | Healthy | Warning |
|---|---|---|
| PPE growth / Revenue growth | ≤1 | >1.5 |
| Capex / Depreciation | >1 normally | Persistently <1 |
| Capex / CFO | <70% | >90% |

Interpretation must consider capital-intensive industries.

---

# 20. Intangibles and Goodwill

| Metric | Formula |
|---|---|
| Intangibles / Assets | Intangibles / Assets |
| Goodwill / Assets | Goodwill / Assets |
| Goodwill / Equity | Goodwill / Equity |
| Goodwill Growth | YoY goodwill growth |
| Acquisition Premium | (Purchase Price − FV Net Assets) / Purchase Price |
| Impairment / Goodwill | Impairment / Goodwill |
| Amortization / Intangibles | Amortization / Intangibles |

### Screening

| Metric | Healthy | Warning |
|---|---|---|
| Goodwill / Assets | <10% | >20–25% |
| Intangibles / Assets | Industry-normal | >30% |

Rapid goodwill growth followed by delayed impairment is particularly important to investigate.

---

# 21. Profitability

| Metric | Formula |
|---|---|
| Gross Margin | Gross Profit / Revenue |
| EBITDA Margin | EBITDA / Revenue |
| EBIT Margin | EBIT / Revenue |
| Operating Margin | Operating Profit / Revenue |
| Net Margin | NI / Revenue |
| ROA | NI / Avg Assets |
| ROE | NI / Avg Equity |
| ROIC | NOPAT / Invested Capital |
| ROCE | EBIT / Capital Employed |

### Key Forensic Rule

There are few universal profitability cutoffs.

Use:

1. Historical company benchmarks
2. Industry peers
3. Business-model expectations
4. Capital intensity
5. Cost of capital

A high ROIC relative to WACC is generally desirable.

---

# 22. Tax Forensics

| Metric | Formula |
|---|---|
| Effective Tax Rate | Tax Expense / PBT |
| Cash Tax Rate | Cash Taxes Paid / PBT |
| Tax Expense / CFO | Tax Expense / CFO |
| DTA / Assets | Deferred Tax Assets / Assets |
| DTL / Assets | Deferred Tax Liabilities / Assets |
| ETR Volatility | Standard deviation of ETR |

### Warning Signs

- Large persistent gap between accounting ETR and cash tax rate.
- Repeated unusually low ETR.
- Rapid growth in deferred tax assets.
- Tax benefits repeatedly driving earnings.

---

# 23. Interest and Financing

| Metric | Formula |
|---|---|
| Interest Expense / Debt | Interest Expense / Avg Debt |
| Interest Expense / Revenue | Interest Expense / Revenue |
| Interest Coverage | EBIT / Interest |
| CFO / Interest | CFO / Interest |
| Debt Due Within 1 Year / Total Debt | Current Maturities / Total Debt |
| Short-Term Debt / Total Debt | STD / Total Debt |

Warning signs include:

- Rising borrowing cost.
- Debt refinancing dependence.
- Large near-term maturities.
- Declining interest coverage.

---

# 24. Cash Flow Forensics

| Metric | Formula |
|---|---|
| CFO / NI | CFO / NI |
| CFO / EBITDA | CFO / EBITDA |
| CFO / Revenue | CFO / Revenue |
| FCF / NI | FCF / NI |
| FCF Margin | FCF / Revenue |
| Capex / CFO | Capex / CFO |
| Financing CF / CFO | Financing CF / CFO |
| Investing CF / CFO | Investing CF / CFO |
| CFO Volatility | σ(CFO) |
| FCF Volatility | σ(FCF) |
| CFO ex-WC | CFO − ΔWC |
| WC Contribution to CFO | ΔWC / CFO |

Also examine whether CFO depends heavily on:

- Increasing payables
- Declining receivables
- Inventory liquidation
- Other temporary working-capital changes

---

# 25. Related-Party Transactions

| Metric | Formula |
|---|---|
| Related-Party Revenue / Revenue | RP Revenue / Revenue |
| RP Receivables / Total AR | RP Receivables / AR |
| RP Loans / Assets | RP Loans / Assets |
| RP Loans / Equity | RP Loans / Equity |
| RP Purchases / COGS | RP Purchases / COGS |
| RP Transactions / Revenue | RP Transactions / Revenue |

Investigate:

- Promoter-owned entities
- Subsidiaries
- Associates
- Joint ventures
- Guarantees
- Loans
- Management fees
- Property transactions
- Inter-company receivables

### Screening

| Metric | Low Risk | Caution | Severe |
|---|---:|---:|---:|
| RP Transactions / Revenue | <10% | 10–20% | >20% |
| RP Receivables / AR | <10% | 10–20% | >20% |
| RP Loans / Equity | <5% | 5–10% | >10% |

These are screening conventions, not universal regulatory thresholds.

---

# 26. Promoter and Insider Analysis

| Metric | Formula |
|---|---|
| Promoter Ownership | Promoter Shares / Total Shares |
| Promoter Pledge Ratio | Pledged Promoter Shares / Promoter Shares |
| Pledge / Total Shares | Pledged Shares / Total Shares |
| Promoter Ownership Change | YoY change |
| Insider Selling / Market Cap | Insider Sales / Market Cap |
| Insider Buying / Market Cap | Insider Purchases / Market Cap |

### Promoter Pledge Screening

| Pledge Ratio | Interpretation |
|---:|---|
| 0% | Good |
| 10–25% | Caution |
| 25–50%+ | Severe |

Again, these are practical screening levels rather than universal thresholds.

---

# 27. Dividend Analysis

| Metric | Formula |
|---|---|
| Dividend Payout | Dividend / NI |
| Dividend Coverage | CFO / Dividend |
| FCF Dividend Coverage | FCF / Dividend |
| Dividend / FCF | Dividend / FCF |
| Borrowing-Funded Dividend | Debt growth while FCF < Dividend |

### Screening

| Dividend / FCF | Interpretation |
|---:|---|
| <70% | Generally comfortable |
| 70–100% | Caution |
| >100% | Potentially unsustainable |

---

# 28. Buybacks

| Metric | Formula |
|---|---|
| Buyback / FCF | Buyback / FCF |
| Buyback / CFO | Buyback / CFO |
| Net Share Count Change | Current shares − prior shares |
| Net Buyback | Repurchases − shares issued |
| EPS Without Buybacks | NI / Prior-period shares |

A company can appear to have strong EPS growth even when underlying earnings growth is weak if share count reduction is doing much of the work.

---

# 29. Stock-Based Compensation

| Metric | Formula |
|---|---|
| SBC / Revenue | SBC / Revenue |
| SBC / Operating Expenses | SBC / OpEx |
| SBC / NI | SBC / NI |
| SBC / CFO | SBC / CFO |
| Dilution | ΔShares / Shares |

### Screening

| Metric | Generally Comfortable | Warning |
|---|---:|---:|
| SBC / Revenue | <2% | >5% |
| SBC / NI | <10% | >25% |
| SBC / CFO | <10% | >25% |
| Share Dilution | <1% | >3–5% |

---

# 30. EPS Forensics

| Metric | Formula |
|---|---|
| EPS Growth | EPS_t / EPS_t-1 − 1 |
| NI Growth | NI_t / NI_t-1 − 1 |
| EPS vs NI Growth | Compare growth rates |
| Basic vs Diluted EPS Gap | Basic EPS − Diluted EPS |
| Share Count Growth | ΔShares / Shares |
| EPS Without Buybacks | NI / Prior Shares |

### Warning Signs

- EPS grows much faster than net income.
- Share count is falling rapidly.
- Buybacks are financed by debt.
- Diluted EPS differs materially from basic EPS.

---

# 31. Non-GAAP Earnings

| Metric | Formula |
|---|---|
| Adjustments / GAAP Earnings | Adjustments / GAAP NI |
| Adjusted EBITDA / GAAP EBITDA | Adjusted EBITDA / GAAP EBITDA |
| Recurring Adjustments / Revenue | Recurring adjustments / Revenue |
| One-Time Expense Frequency | Count of repeated "one-time" adjustments |

### Warning Signs

- Repeated expenses labeled "one-time."
- Adjusted earnings persistently much higher than GAAP earnings.
- Acquisition, restructuring, stock compensation, or other exclusions recurring every year.

---

# 32. Acquisition Forensics

| Metric | Formula |
|---|---|
| Acquisition Spend / CFO | Acquisition Cash Outflow / CFO |
| Acquisition Spend / FCF | Acquisition Cash Outflow / FCF |
| Goodwill Created / Acquisition Price | Goodwill / Purchase Price |
| Acquisition Growth Contribution | Acquired Revenue / Total Revenue |
| Organic Growth | Reported Growth − Acquisition Growth |
| Acquisition-Adjusted ROIC | ROIC excluding acquisition effects |

### Screening

| Test | Generally Comfortable | Warning |
|---|---:|---:|
| Acquisition / CFO | <30% | >50% |
| Goodwill / Purchase Price | Low | >50% |

---

# 33. Segment Analysis

| Metric | Formula |
|---|---|
| Segment Revenue Growth | YoY segment revenue growth |
| Segment Margin | Segment Profit / Segment Revenue |
| Segment Asset Turnover | Segment Revenue / Segment Assets |
| Segment ROIC | Segment NOPAT / Segment Invested Capital |
| Segment Concentration | Largest segment / Total Revenue |

Look for:

- One segment driving nearly all profits.
- Unusual margin expansion.
- Segment losses hidden by consolidated earnings.
- Rapid growth in low-quality or low-cash-conversion segments.

---

# 34. Off-Balance-Sheet Risk

| Metric | Formula |
|---|---|
| Guarantees / Equity | Guarantees / Equity |
| Guarantees / Assets | Guarantees / Assets |
| Contingent Liabilities / Equity | Contingent Liabilities / Equity |
| Contingent Liabilities / Assets | Contingent Liabilities / Assets |
| Lease Liabilities / Debt | Lease Liabilities / Debt |
| Adjusted Debt | Debt + Lease Liabilities |

### Screening

| Ratio | Good | Caution | Severe |
|---|---:|---:|---:|
| Contingent Liabilities / Equity | <10% | 10–25% | >25% |
| Guarantees / Equity | <10% | 10–25% | >25% |

---

# 35. Balance-Sheet Quality

| Metric | Formula |
|---|---|
| Cash / Assets | Cash / Assets |
| AR / Assets | AR / Assets |
| Inventory / Assets | Inventory / Assets |
| Other Current Assets / Assets | OCA / Assets |
| Other Assets / Assets | Other Assets / Assets |
| Other Liabilities / Assets | Other Liabilities / Assets |
| Equity / Assets | Equity / Assets |

### Warning Signs

- Other assets growing faster than the business.
- Receivables increasing faster than revenue.
- Inventory increasing faster than sales.
- Cash falling while reported earnings rise.
- Large unexplained changes in other assets/liabilities.

---

# 36. Margin Anomaly Tests

| Metric | Formula |
|---|---|
| Δ Gross Margin | GM_t − GM_t-1 |
| Δ EBITDA Margin | EBITDA Margin_t − EBITDA Margin_t-1 |
| Δ EBIT Margin | EBIT Margin_t − EBIT Margin_t-1 |
| Δ Net Margin | Net Margin_t − Net Margin_t-1 |
| Δ CFO Margin | CFO Margin_t − CFO Margin_t-1 |

Compare:

\[
\Delta EBITDA\ Margin
\]

against:

\[
\Delta Gross\ Margin
\]

and:

\[
\Delta CFO\ Margin
\]

A sharp profit-margin improvement without corresponding cash-flow improvement deserves investigation.

---

# 37. Earnings Smoothing

| Test | Formula |
|---|---|
| NI Volatility | σ(NI) |
| CFO Volatility | σ(CFO) |
| Accrual Volatility | σ(NI − CFO) |
| NI-CFO Correlation | Corr(NI,CFO) |
| Revenue-NI Correlation | Corr(Revenue,NI) |

### Practical Screening

| Test | Stronger Profile | Warning |
|---|---:|---:|
| Corr(NI,CFO) | >0.70 | <0.30 |
| NI volatility vs CFO volatility | Similar | NI unusually smooth |
| NI-CFO gap | Stable | Persistent widening |

---

# 38. Benford's Law

Expected first-digit probability:

\[
P(d)=\log_{10}\left(1+\frac{1}{d}\right)
\]

for:

\[
d=1,\ldots,9
\]

Possible tests:

- First digit
- First two digits
- MAD
- Chi-square
- Jensen-Shannon divergence

### MAD Screening

| MAD | Interpretation |
|---:|---|
| <0.006 | Close conformity |
| 0.006–0.012 | Acceptable / questionable |
| 0.012–0.015 | Marginal |
| >0.015 | Non-conformity |

**Important:** Benford deviations are not evidence of fraud by themselves. They may result from natural data-generation processes, business size, account structure, or small samples.

---

# 39. Market-Based Ratios

| Metric | Formula |
|---|---|
| Price / Sales | Market Cap / Revenue |
| Price / Book | Market Cap / Equity |
| EV / Sales | Enterprise Value / Revenue |
| EV / EBITDA | EV / EBITDA |
| EV / EBIT | EV / EBIT |
| Price / FCF | Market Cap / FCF |
| Earnings Yield | EPS / Price |
| FCF Yield | FCF / Market Cap |
| EV / FCF | EV / FCF |
| Market Cap / Net Assets | Market Cap / Net Assets |

These are primarily valuation metrics, but can provide forensic context when valuation depends heavily on aggressive earnings assumptions.

---

# 40. Trend and Statistical Analysis

| Metric | Formula |
|---|---|
| 1Y Change | X_t / X_t-1 − 1 |
| YoY Growth | Current / Previous − 1 |
| 3Y CAGR | (X_t/X_t-3)^(1/3) − 1 |
| 5Y CAGR | (X_t/X_t-5)^(1/5) − 1 |
| Trend Slope | Regression slope |
| Trend R² | Regression R² |
| Volatility | Standard deviation |
| Peer Z-Score | (Company − Peer Median) / MAD |
| Percentile Rank | Rank within peer group |
| Velocity | X_t − X_t-1 |
| Acceleration | (X_t−X_t-1) − (X_t-1−X_t-2) |

### Peer Z-Score

\[
PeerZ =
\frac{CompanyMetric-PeerMedian}
{MAD}
\]

### Practical Interpretation

| Peer Z | Interpretation |
|---:|---|
| -2 to +2 | Generally normal |
| >±2 | Unusual |
| >±3 | Extreme |

### Percentile

| Percentile | Interpretation |
|---:|---|
| 25–75 | Generally normal |
| >90 | Extreme high |
| <10 | Extreme low |

---

# 41. Consolidated Benchmark Table

| Model / Test | Green / Healthy | Yellow / Caution | Red / High Risk | Benchmark Type |
|---|---|---|---|---|
| Beneish M-Score | < -1.78 | — | > -1.78 | Academic screening |
| Dechow F-Score | <1 | ≥1 | ≥1.85 / ≥2.45 | Published/conventional |
| Piotroski F-Score | 8–9 | 5–7 | 0–4 | Model score |
| Altman Z | >2.99 | 1.81–2.99 | <1.81 | Academic model |
| Altman Z″ | >2.60 | 1.10–2.60 | <1.10 | Variant-dependent |
| Sloan Accrual | <5% | 5–10% | >10% | Screening |
| Abnormal Accruals | Around 0 | Moderate | Large positive | Peer/industry |
| CFO / NI | >1.0 | 0.7–1.0 | <0.7 | Screening |
| FCF / NI | >0.8 | 0–0.8 | <0 | Screening |
| CFO / EBITDA | >0.8 | 0.6–0.8 | <0.6 | Screening |
| CFO / EBIT | >1.0 | 0.7–1.0 | <0.7 | Screening |
| Accrual / Assets | <3% | 3–5% | >5% | Screening |
| Current Ratio | >1.5 | 1–1.5 | <1 | Screening |
| Quick Ratio | >1 | 0.7–1 | <0.7 | Screening |
| Cash Ratio | >0.5 | 0.2–0.5 | <0.2 | Screening |
| Debt / Equity | <1 | 1–2 | >2 | Screening |
| Debt / Assets | <40% | 40–60% | >60% | Screening |
| Net Debt / EBITDA | <2 | 2–3 | >3 | Screening |
| Debt / CFO | <3 | 3–5 | >5 | Screening |
| Interest Coverage | >5 | 2–5 | <2 | Screening |
| Capex / CFO | <70% | 70–90% | >90% | Screening |
| Dividend / FCF | <70% | 70–100% | >100% | Screening |
| Buyback / FCF | <70% | 70–100% | >100% | Screening |
| Goodwill / Assets | <10% | 10–20% | >20–25% | Screening |
| SBC / Revenue | <2% | 2–5% | >5% | Screening |
| SBC / NI | <10% | 10–25% | >25% | Screening |
| Share Dilution | <1% | 1–3% | >3–5% | Screening |
| Promoter Pledge | 0% | 10–25% | >25–50% | Screening |
| RP Transactions / Revenue | <10% | 10–20% | >20% | Screening |
| RP Loans / Equity | <5% | 5–10% | >10% | Screening |
| Contingent Liabilities / Equity | <10% | 10–25% | >25% | Screening |
| Benford MAD | <0.006 | 0.006–0.015 | >0.015 | Statistical screening |
| Peer Z-Score | -2 to +2 | ±2 to ±3 | >±3 | Statistical screening |
| Corr(NI,CFO) | >0.70 | 0.30–0.70 | <0.30 | Screening |

---

# 42. Benchmark Classification Hierarchy

Not every benchmark has the same evidentiary status.

## Tier 1 — Academic / Model-Validated Thresholds

Use these as the strongest benchmark signals:

- Beneish M-Score threshold
- Original Altman Z zones
- Piotroski 0–9 score
- Published Dechow F-Score interpretation

## Tier 2 — Statistical / Peer Benchmarks

Best for:

- Jones abnormal accruals
- Modified Jones
- Kasznik
- Peer Z-scores
- Percentile ranks
- Industry-adjusted ratios

## Tier 3 — Practical Forensic Screening Benchmarks

Useful for:

- CFO/NI
- DSO changes
- Inventory growth
- Debt ratios
- Related-party exposure
- Promoter pledge
- SBC
- Capex/CFO
- Dividend/FCF

These should **not** be treated as universal accounting rules.

## Tier 4 — Company-Specific Benchmarks

For high-quality forensic analysis, compare each company with:

1. Its own 5–10 year history
2. Direct competitors
3. Industry median
4. Industry upper/lower quartiles
5. Business-cycle conditions
6. Accounting-policy changes

---

# 43. Recommended Forensic Scoring Architecture

A robust automated system should not depend on a single model.

A useful architecture is:

\[
ForensicRiskScore =
w_1(MScore)
+w_2(FScore)
+w_3(AccrualRisk)
+w_4(CashFlowRisk)
+w_5(BalanceSheetRisk)
+w_6(GovernanceRisk)
+w_7(WorkingCapitalRisk)
+w_8(StatisticalAnomaly)
\]

Each category can be normalized to a 0–100 risk score.

## Example

| Category | Weight |
|---|---:|
| Beneish / Earnings Manipulation | 20% |
| Dechow / Fraud Risk | 15% |
| Accrual Quality | 10% |
| Cash-Flow Quality | 15% |
| Balance-Sheet Quality | 10% |
| Leverage / Solvency | 10% |
| Working Capital | 5% |
| Governance / Related Parties | 10% |
| Statistical Anomalies | 5% |
| **Total** | **100%** |

These weights are a proposed implementation framework, not an academically validated universal weighting scheme.

---

# 44. Recommended Risk Levels

| Composite Score | Risk Level |
|---:|---|
| 0–20 | Very Low |
| 20–40 | Low |
| 40–60 | Moderate |
| 60–75 | High |
| 75–90 | Very High |
| 90–100 | Critical |

The scoring engine should retain the underlying component scores so the analyst can explain **why** the composite score is high.

---

# 45. Red-Flag Escalation Logic

A single red flag should generally trigger investigation rather than automatically declaring fraud.

A stronger forensic signal occurs when multiple independent indicators agree.

### Example High-Risk Cluster

If a company simultaneously shows:

- Beneish M > -1.78
- Dechow F ≥ 1
- CFO/NI < 0.7
- DSO rising >15%
- AR growth > revenue growth
- Inventory growth > sales growth
- Accrual/assets >5%
- Related-party receivables rising
- Promoter pledge increasing
- Goodwill rising rapidly
- Persistent "one-time" adjustments

then the combined risk is materially more significant than any individual indicator.

---

# 46. Forensic Analysis Principles

## Principle 1 — Follow the Cash

Reported profit is less reliable when:

\[
CFO \ll NI
\]

for multiple years.

## Principle 2 — Follow Receivables

Rapid AR growth relative to revenue can indicate:

- Aggressive revenue recognition
- Channel stuffing
- Weak collections
- Extended credit terms
- Contract modifications

## Principle 3 — Follow Working Capital

Cash flow supported primarily by:

- Higher payables
- Lower receivables
- Inventory liquidation

may not be sustainable.

## Principle 4 — Follow Related Parties

Related-party activity should be assessed for:

- Pricing
- Receivables
- Loans
- Guarantees
- Asset transfers
- Management fees
- Circular transactions

## Principle 5 — Follow Equity and Debt

Strong EPS can be misleading if driven by:

- Buybacks
- Debt-funded distributions
- Excessive leverage
- Share-count reduction

## Principle 6 — Follow Accounting Policies

Focus on:

- Revenue recognition
- Capitalization policies
- Depreciation assumptions
- Useful lives
- Impairment testing
- Provisions
- Deferred taxes
- Non-GAAP adjustments

---

# 47. Minimum Data Requirements

For a strong forensic system, collect at least:

### Income Statement

- Revenue
- COGS
- Gross profit
- SG&A
- EBITDA
- EBIT
- Interest
- PBT
- Tax
- Net income

### Balance Sheet

- Cash
- Marketable securities
- AR
- Inventory
- Other current assets
- PPE
- Intangibles
- Goodwill
- Total assets
- AP
- Current liabilities
- Long-term debt
- Total liabilities
- Equity

### Cash Flow Statement

- CFO
- Capex
- Investing CF
- Financing CF
- Dividends
- Buybacks

### Other Data

- Shares outstanding
- Diluted shares
- Promoter ownership
- Promoter pledge
- Insider transactions
- Related-party transactions
- Contingent liabilities
- Guarantees
- Lease liabilities
- Acquisitions
- Segment data
- Auditor information
- Accounting policy changes

---

# 48. Recommended Historical Window

For automated forensic analysis:

| Analysis | Minimum | Preferred |
|---|---:|---:|
| Beneish | 2 years | 3–5 years |
| Dechow | 2 years | 3–5 years |
| Piotroski | 2 years | 3–5 years |
| Altman | 1 year | 3–5 years |
| Accrual analysis | 3 years | 5–10 years |
| Working capital | 3 years | 5–10 years |
| Related-party | 3 years | 5–10 years |
| Benford | Sufficient transactions | Large transaction population |
| Trend analysis | 3 years | 5–10 years |

---

# 49. Important Implementation Warnings

1. **Do not treat any single forensic model as proof of fraud.**
2. **Do not apply universal ratio thresholds blindly across industries.**
3. Banks and insurers require specialized balance-sheet and regulatory analysis.
4. Commodity companies require cycle-adjusted benchmarks.
5. High-growth companies naturally have different working-capital profiles.
6. Capital-intensive industries naturally have different capex ratios.
7. SaaS companies naturally have high deferred revenue and different cash-conversion dynamics.
8. Conglomerates require segment-level analysis.
9. Acquisition-heavy businesses require acquisition-adjusted metrics.
10. Benford's Law requires an appropriate transaction population.
11. Jones-family models are most meaningful when estimated against appropriate industry-year peer groups.
12. Altman variants must match the company's business type.
13. Beneish component thresholds are screening conventions rather than official individual cutoffs.
14. Related-party thresholds are context-dependent.
15. Promoter pledge thresholds are screening conventions.
16. The final forensic conclusion should combine quantitative and qualitative evidence.

---

# 50. Final Forensic Decision Framework

A high-quality stock forensic engine should produce four separate outputs:

## A. Manipulation Risk

Driven by:

- Beneish M-Score
- Dechow F-Score
- Jones / Modified Jones / Kasznik
- Accrual ratios
- Revenue-quality indicators
- Benford anomalies

## B. Financial Distress Risk

Driven by:

- Altman Z variants
- Debt/EBITDA
- Debt/CFO
- Interest coverage
- Current ratio
- Cash ratio
- DSCR

## C. Earnings Quality

Driven by:

- CFO/NI
- FCF/NI
- CFO/EBITDA
- Accrual intensity
- Working-capital dependence
- EPS/share-count analysis

## D. Governance / Structural Risk

Driven by:

- Promoter pledge
- Related-party transactions
- Insider activity
- Guarantees
- Contingent liabilities
- Subsidiaries/JVs
- Auditor changes
- Accounting-policy changes
- Non-GAAP adjustments

---

# 51. Ideal Final Output for Any Listed Stock

The final automated report should contain:

1. **Overall Forensic Risk Score**
2. **Manipulation Risk Score**
3. **Financial Distress Score**
4. **Earnings Quality Score**
5. **Balance-Sheet Quality Score**
6. **Cash-Flow Quality Score**
7. **Governance Risk Score**
8. **Working-Capital Risk Score**
9. **Capital Allocation Score**
10. **Statistical Anomaly Score**
11. **Top 10 Red Flags**
12. **Top 10 Positive Signals**
13. **Model-by-model results**
14. **Historical trend charts**
15. **Peer comparison**
16. **Industry-adjusted scores**
17. **Management / promoter analysis**
18. **Related-party analysis**
19. **Accounting-policy analysis**
20. **Final forensic conclusion**

The most useful final conclusion should distinguish:

- **No major forensic concerns**
- **Some areas require monitoring**
- **Elevated forensic risk**
- **High forensic risk**
- **Critical forensic risk requiring deep investigation**

The system should explain the evidence behind the classification rather than simply outputting a numerical score.
