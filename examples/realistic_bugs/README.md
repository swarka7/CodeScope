# Realistic Bug Benchmarks

These projects are independent benchmark applications for validating repository analysis and
debugging tools on realistic Python code. They are intentionally small, layered applications with
normal pytest tests and one known failing behavior each.

They are not production systems and are not proof of production readiness. They are small benchmark
apps designed to check whether CodeScope can surface useful debugging context for realistic
multi-file failures.

Run all benchmark tests from the repository root:

```powershell
python -m pytest examples/realistic_bugs
```

The command is expected to fail because each app contains one intentional business-logic bug.
The expected current result is `3 failed, 9 passed`.

Run CodeScope diagnosis on each benchmark app:

```powershell
python -m codescope.cli index examples/realistic_bugs/banking_app
python -m codescope.cli diagnose examples/realistic_bugs/banking_app

python -m codescope.cli index examples/realistic_bugs/movie_platform
python -m codescope.cli diagnose examples/realistic_bugs/movie_platform

python -m codescope.cli index examples/realistic_bugs/inventory_app
python -m codescope.cli diagnose examples/realistic_bugs/inventory_app
```

Run the benchmark evaluator:

```powershell
python -m codescope.cli benchmark examples/realistic_bugs
```

Expected evaluator result:

```text
3 PASS, 0 PARTIAL, 0 FAIL
```

Run natural-language investigation without pytest:

```powershell
python -m codescope.cli investigate examples/realistic_bugs/banking_app "When I transfer money, the receiver balance does not increase"
python -m codescope.cli investigate examples/realistic_bugs/movie_platform "When I filter movies by genre and rating, wrong genres appear"
python -m codescope.cli investigate examples/realistic_bugs/inventory_app "When I ship an order with insufficient stock, it is allowed"
```

Use JSON output for tools and future editor integrations:

```powershell
python -m codescope.cli diagnose examples/realistic_bugs/banking_app --json
python -m codescope.cli investigate examples/realistic_bugs/banking_app "When I transfer money, the receiver balance does not increase" --json
```

## Banking Transfer App

Path: `examples/realistic_bugs/banking_app`

Domain: a simple account-transfer workflow with models, in-memory persistence, a business service,
and route-style functions.

Expected failing test:

```powershell
python -m pytest examples/realistic_bugs/banking_app
```

Business bug: a successful transfer debits the sender and records the transfer, but the receiver's
balance is not credited.

Expected CodeScope root-cause chunk: `TransferService.transfer`.

## Movie Rating/Search Platform

Path: `examples/realistic_bugs/movie_platform`

Domain: a movie catalog with reviews, aggregate ratings, search criteria, and route-style query
handling.

Expected failing test:

```powershell
python -m pytest examples/realistic_bugs/movie_platform
```

Business bug: combined search criteria do not apply the genre filter, so highly rated movies from
other genres can appear in results.

Expected CodeScope root-cause chunk: `MovieSearchService.search`.

## Inventory/Order App

Path: `examples/realistic_bugs/inventory_app`

Domain: a basic order fulfillment workflow with products, orders, repositories, service logic,
policy checks, and route-style functions.

Expected failing test:

```powershell
python -m pytest examples/realistic_bugs/inventory_app
```

Business bug: shipping an order does not check available stock before reserving inventory, allowing
orders to ship with insufficient stock.

Expected CodeScope root-cause chunk: `FulfillmentService.ship_order`.
