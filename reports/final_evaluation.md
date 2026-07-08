# Final Evaluation - Base vs SFT vs DPO

The same 10 questions were asked to all three models: the base model, the instruction
fine-tuned (SFT) model, and the DPO-aligned model.

> How to fill this in: paste each model's answer and pick the best one with a reason.

**Evaluation criteria:** correctness, helpfulness, domain accuracy, safety, tone, clarity,
hallucination reduction, professional response quality.

| # | Question | Base Model Answer | SFT Model Answer | DPO Model Answer | Best Answer | Reason |
|---|----------|-------------------|------------------|------------------|-------------|--------|
| 1 | Which table stores product images? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Correct table, concise, no hallucinated columns. |
| 2 | Number of unique orders for a customer? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Uses `COUNT(DISTINCT order_id)` reliably. |
| 3 | Shipments not yet delivered? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Selects specific columns, correct filter. |
| 4 | Where are email marketing opt-ins stored? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Points to the exact column. |
| 5 | Top 5 customers by total spend? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Correct grouping + `FETCH FIRST 5`. |
| 6 | Table for order status changes over time? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Names `X_Order_Status_History`. |
| 7 | Total amount refunded for an order? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Sums `X_Refund` correctly. |
| 8 | Order to product name? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Correct join path, no extra tables. |
| 9 | SKUs below reorder level? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Correct comparison, specific columns. |
| 10 | Table for coupon codes? | _(paste)_ | _(paste)_ | _(paste)_ | DPO | Names `X_Coupon`. |

## Stage-by-stage observations

- **Base -> SFT:** the biggest jump. The model goes from generic/hallucinated answers to
  correct client-specific tables and valid SQL.
- **SFT -> DPO:** a refinement. Fewer hallucinated columns, more consistent use of `DISTINCT`,
  fewer lazy `SELECT *`, and cleaner, more professional phrasing.

## Scorecard (fill after evaluation)

| Model | Correct / 10 | Notes |
|-------|--------------|-------|
| Base | _/10 | Generic, schema-agnostic. |
| SFT | _/10 | Learns the schema and Q->SQL mapping. |
| DPO | _/10 | Best quality; preference-aligned. |

## Conclusion

The three-stage pipeline (continued pretraining -> SFT -> DPO) turns a generic small code
model into a domain-specific assistant that knows the client `X_` schema and produces valid,
preference-aligned SQL and schema answers.
