# Realistic Bug Benchmarks

These projects are independent benchmark applications for validating repository analysis and
debugging tools on realistic Python code. They are intentionally small, layered applications with
normal pytest tests and one known failing behavior each.

Run all benchmark tests from the repository root:

```powershell
python -m pytest examples/realistic_bugs
```

The command is expected to fail because each app contains one intentional business-logic bug.

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
