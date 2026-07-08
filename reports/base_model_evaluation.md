# Base Model Evaluation

Before any fine-tuning, the base model (`Qwen2.5-Coder-1.5B`) was tested on 10
domain-specific questions. The goal is to establish a baseline and show that the base model
does not know the client's `X_` schema and gives generic or incorrect answers.

> How to fill this in: run the questions through the base model (the "sanity test" style cell,
> or `inference.py` pointed at the base model) and paste the raw answers into the table.
> The "Problem" column notes why the answer is inadequate.

| # | Question | Base Model Answer | Problem |
|---|----------|-------------------|---------|
| 1 | Which table stores product images? | _(paste)_ | Does not know the schema; guesses a generic name instead of `X_Product_Images`. |
| 2 | How can I find the number of unique orders for a customer? | _(paste)_ | Generic SQL; may forget `DISTINCT` or use a non-existent table. |
| 3 | Give me a query to list all shipments that have not been delivered. | _(paste)_ | Unaware of `X_Shipment.shipment_status`; invents columns. |
| 4 | Where are customer email marketing opt-ins stored? | _(paste)_ | Does not know `X_Customer_Preference.opt_in_email`. |
| 5 | Write a query for the top 5 customers by total spend. | _(paste)_ | May not use `X_Order.total_amount` or correct grouping. |
| 6 | Which table records order status changes over time? | _(paste)_ | Unaware of `X_Order_Status_History`. |
| 7 | Get the total amount refunded for a given order. | _(paste)_ | Does not know `X_Refund`; may look in the wrong table. |
| 8 | How do I get from an order to the product name? | _(paste)_ | Misses the `X_Order_Item -> X_SKU -> X_Product` join path. |
| 9 | Find all SKUs below their reorder level. | _(paste)_ | Unaware of `X_Inventory.reorder_level`. |
| 10 | Which table stores coupon codes? | _(paste)_ | Does not know `X_Coupon`. |

## Summary

The base model produces **generic, schema-agnostic** answers. It has no knowledge of the
client's `X_`-prefixed tables, so it either hallucinates table/column names or gives
textbook SQL that would not run against this database. This motivates the three-stage
fine-tuning that follows.
