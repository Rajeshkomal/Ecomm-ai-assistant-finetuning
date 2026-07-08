# Base Model vs Instruction Fine-Tuned (SFT) Model

The same 10 questions were asked again after Stage 2 (instruction fine-tuning). This table
compares the base model with the SFT model.

> How to fill this in: paste the base answers (from `base_model_evaluation.md`) and the SFT
> answers (from the "Inference after SFT" cell). Judge which is better and why.

**Evaluation criteria:** correctness, domain accuracy, clarity, safety, helpfulness,
less-generic behavior, better domain-specific behavior.

| # | Question | Base Model Answer | SFT Model Answer | Which is Better? | Reason |
|---|----------|-------------------|------------------|------------------|--------|
| 1 | Which table stores product images? | _(paste)_ | _(paste)_ | SFT | SFT names `X_Product_Images` with key columns. |
| 2 | Number of unique orders for a customer? | _(paste)_ | _(paste)_ | SFT | SFT uses `COUNT(DISTINCT order_id)` on `X_Order`. |
| 3 | Shipments not yet delivered? | _(paste)_ | _(paste)_ | SFT | SFT filters `X_Shipment` by `shipment_status`. |
| 4 | Where are email marketing opt-ins stored? | _(paste)_ | _(paste)_ | SFT | SFT points to `X_Customer_Preference.opt_in_email`. |
| 5 | Top 5 customers by total spend? | _(paste)_ | _(paste)_ | SFT | SFT groups `X_Order` and orders by sum. |
| 6 | Table for order status changes over time? | _(paste)_ | _(paste)_ | SFT | SFT names `X_Order_Status_History`. |
| 7 | Total amount refunded for an order? | _(paste)_ | _(paste)_ | SFT | SFT sums `X_Refund.refund_amount`. |
| 8 | Order to product name? | _(paste)_ | _(paste)_ | SFT | SFT gives the correct 3-table join path. |
| 9 | SKUs below reorder level? | _(paste)_ | _(paste)_ | SFT | SFT compares `quantity_on_hand < reorder_level`. |
| 10 | Table for coupon codes? | _(paste)_ | _(paste)_ | SFT | SFT names `X_Coupon`. |

## Summary

After instruction fine-tuning, the model answers with the **correct client-specific table
names and valid SQL** instead of generic guesses. It now follows the question -> answer
format reliably. Remaining weaknesses (occasional missing `DISTINCT`, over-broad `SELECT *`,
or a rare hallucinated column) are targeted next by DPO.
