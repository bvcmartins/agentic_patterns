# NPV / IRR / payback conventions for ranking competing projects

Use this for project/asset capital allocation where cash flows are the inputs
(not risk-weighted returns — for those see the `raroc` skill).

## Cash-flow conventions (get these right or every metric is wrong)

- **Sign:** outflows negative, inflows positive.
- **t=0:** `cashflows[0]` is the initial outlay at time zero and is NOT discounted.
  Period 1's flow is `cashflows[1]`, discounted once, etc.
- **Periodicity:** the discount rate must match the cash-flow period. Annual flows
  → annual rate. Monthly flows → monthly rate (`annual/12`).
- **Nominal vs. real:** discount nominal flows at a nominal rate, real at real.
  Don't mix.

## Metrics and the `calc` calls

    NPV at hurdle rate r:   calc("npv(0.10, [-1000, 300, 400, 500, 600])")
    IRR:                    calc("irr([-1000, 300, 400, 500, 600])")
    Profitability Index:    PI = (NPV + |initial outlay|) / |initial outlay|
                            calc("(npv(0.10, cf) + 1000) / 1000")   # outlay = 1000
    Discounted payback:     accumulate discounted flows until cumulative ≥ 0;
                            report the period it crosses.

## Decision rules

- **NPV > 0** at the hurdle rate → creates value → fundable.
- **IRR > hurdle** → fundable (same direction as NPV for conventional flows).
- Under a **fixed capital budget**, rank by **Profitability Index** (value per
  unit of capital), not by raw NPV — raw NPV favours large projects regardless
  of how much capital they tie up.

## Pitfalls

- **Non-conventional flows** (sign changes more than once) can produce multiple
  or no IRR — trust NPV, and note the ambiguity. The ported `irr()` brackets and
  bisects, falling back to Newton; it returns one root.
- **IRR ≠ ranking tool across mutually exclusive projects** of different scale or
  timing — NPV (or PI under a budget) is the correct tie-breaker.
- **Mismatched horizons.** Comparing a 3-year and a 10-year project on NPV alone
  is fine; on IRR or payback it can mislead. Equivalent Annual Annuity helps when
  projects repeat.
- **Where did the hurdle rate come from?** Always cite it (warehouse reference
  table vs. stated assumption) — it is the single most decision-sensitive input.

## Warehouse workflow

1. `get_table_info` to confirm the cash-flow table grain (one row per
   project×period?) and the project id / period / amount columns.
2. Pull and order flows per project in SQL:

       SELECT project_id, period, SUM(amount) AS cf
       FROM `project.dataset.project_cashflows`
       WHERE scenario = 'base'
       GROUP BY project_id, period
       ORDER BY project_id, period

3. Assemble each project's ordered list (t=0 first) and compute NPV/IRR/PI with
   `calc`. Rank, then recommend an allocation that maximises total NPV within the
   stated budget, noting what is funded vs. deferred.
