# RAROC — risk-adjusted return on capital for ranking lending/business units

Use this when the capital-allocation question involves **risk-weighted** returns
(banks, lending books, insurance, trading desks) rather than plain project NPV.
RAROC puts units with very different risk profiles on one comparable basis.

## Definition

    RAROC = (risk-adjusted net income) / (economic capital)

Risk-adjusted net income for a lending book is typically:

    revenue
      − funding cost
      − operating expense
      − Expected Loss (EL)        <- the key risk adjustment
      (+/− tax, if you compare after-tax)

where, per exposure:

    EL = PD × LGD × EAD
      PD  = probability of default (1y)
      LGD = loss given default (fraction, 0..1)
      EAD = exposure at default (currency)

Economic capital (the denominator) is the buffer for **Unexpected Loss** at the
chosen confidence level — usually supplied by a risk/capital reference table
(`economic_capital`, `rwa`, `capital_charge`). If only Risk-Weighted Assets are
available, approximate `economic_capital ≈ capital_ratio × RWA` and state the
`capital_ratio` assumption explicitly.

## Decision rule

Compare RAROC to the **hurdle rate** (the firm's cost of equity / target return):

    RAROC > hurdle  -> creates value, candidate to fund
    RAROC < hurdle  -> destroys value, candidate to shrink/deprioritise

Equivalent EVA-style spread:  `economic_profit = (RAROC − hurdle) × economic_capital`.
Rank by economic_profit when you must allocate a fixed capital budget — it is
value created per unit, scaled by the capital each unit consumes.

## How to pull it from the warehouse

1. `get_table_info` on the exposure/loan table — confirm columns for PD, LGD,
   EAD (or balance), revenue, and the segment/unit id and date grain.
2. Find the capital reference: a table with `economic_capital` / `rwa` /
   `capital_charge` per unit, and a `cost_of_capital` / `hurdle_rate` table.
3. Aggregate in SQL per unit (do NOT pull raw rows):

       SELECT unit_id,
              SUM(revenue)                          AS revenue,
              SUM(funding_cost)                     AS funding_cost,
              SUM(opex)                             AS opex,
              SUM(pd * lgd * ead)                   AS expected_loss,
              SUM(economic_capital)                 AS econ_capital
       FROM `project.dataset.exposures`
       WHERE as_of_date = DATE '2025-12-31'
       GROUP BY unit_id

4. Compute per unit with `calc`:

       raroc = (revenue − funding_cost − opex − expected_loss) / econ_capital
       calc("(revenue - funding_cost - opex - expected_loss) / econ_capital")

5. Rank by RAROC and by economic_profit; recommend funding the highest spreads
   first until the capital budget is exhausted.

## Gotchas

- **Consistent horizon & confidence.** EL is a 1-year expected number; economic
  capital must be at the same horizon/confidence as the firm's hurdle. Mismatches
  silently distort the ranking — flag if you can't confirm them.
- **Gross vs. net revenue.** Confirm whether `revenue` already nets funding cost;
  double-subtracting is a common error.
- **Currency.** Normalise all amounts to one currency before dividing.
- **Capital floors / diversification.** Stand-alone economic capital ignores
  portfolio diversification; if the warehouse has a diversified capital figure,
  prefer it and say so.
- **Hurdle source.** Always state where the hurdle rate came from (a reference
  table vs. a stated assumption) — it flips fund/defer decisions near the margin.
