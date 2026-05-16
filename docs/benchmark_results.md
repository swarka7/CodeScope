# CodeScope v0.1 Benchmark Results

CodeScope uses small realistic benchmark applications to check whether failure-aware retrieval surfaces useful debugging context.
These benchmarks are intentionally broken and are not part of the main passing test suite.

## Success Rule

- **PASS**: expected root-cause chunk appears in the top 3 likely relevant code results.
- **PARTIAL**: expected root-cause chunk appears in the top 5 likely relevant code results.
- **FAIL**: expected root-cause chunk is missing from the top 5 likely relevant code results.

## Summary

| app name | bug description | failing test | expected root-cause chunk | observed rank | result |
| --- | --- | --- | --- | --- | --- |
| `banking_app` | transfer debits sender but does not credit receiver | `test_successful_transfer_moves_money_and_records_activity` | `TransferService.transfer` | 1 | PASS |
| `movie_platform` | combined search filters ignore genre | `test_combined_filters_require_genre_rating_and_year_to_match` | `MovieSearchService.search` | 1 | PASS |
| `inventory_app` | insufficient-stock shipment is allowed | `test_order_with_insufficient_stock_cannot_ship` | `FulfillmentService.ship_order` | 3 | PASS |

## Banking App

**Bug:** a successful transfer decreases the sender balance but does not increase the receiver balance.

**What CodeScope found:** `TransferService.transfer` ranked first. CodeScope also surfaced `Account.credit` as paired state-operation context, which is useful because the missing operation is the receiver-side credit.

**Why the output is useful:** the top result points to the business workflow where the omission happens, while related context shows the counterpart operation that exists but is not called.

**Limitation:** CodeScope retrieves the likely omission site and counterpart operation, but it does not prove the missing call or generate a fix.

## Movie Platform

**Bug:** combined search filters ignore the requested genre, so results that match rating/year but not genre are returned.

**What CodeScope found:** `MovieSearchService.search` ranked first.

**Why the output is useful:** the top result is the filtering logic rather than the route or repository plumbing. This is the right layer for a search/filter assertion failure.

**Limitation:** CodeScope does not currently produce a structured explanation of which individual filter criterion is missing.

## Inventory App

**Bug:** an order with insufficient stock is allowed to ship.

**What CodeScope found:** `OutOfStockError` and `require_available_stock` ranked highly, and `FulfillmentService.ship_order` appeared at rank 3.

**Why the output is useful:** the diagnosis connects the expected rejection path to the business operation that should enforce it.

**Limitation:** CodeScope retrieves the validator and caller, but it does not yet explicitly state that the shipping operation likely missed a validator call.

## Honest Limitations

- Benchmarks are intentionally small.
- These results are not proof of production readiness.
- Rankings may change with embedding model, embedding-text version, or scoring changes.
- Benchmark apps intentionally contain failing tests.
- CodeScope retrieves likely debugging context; it does not fix code automatically.
