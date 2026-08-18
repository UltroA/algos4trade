# Security policy

## Scope

This is a research/educational benchmark suite with **read-only** market-data
access (see "Limitations" in README.md) - it never places orders and never
writes to any brokerage account. The main thing worth securing here is the
T-Invest API token used to fetch historical candles.

## Handling the T-Invest token

- Always use a **read-only** T-Invest token (see README section 1). This
  project has no code path that needs trading permissions, and none should
  be granted.
- The token lives in `.env`, which is gitignored - never commit it, never
  paste it into a commit message, and never include it in a script argument
  that might end up in shell history you share.
- **When filing an issue or PR, or pasting logs/output anywhere public: check
  for `T_INVEST_TOKEN`, `Authorization: Bearer ...`, or any string starting
  with `t.` before posting.** Error tracebacks from `core/providers/tinvest.py`
  do not include the token itself, but a pasted `.env` file or a copied shell
  command (`T_INVEST_TOKEN=t.xxx python ...`) would.
- If a token is ever accidentally exposed (committed, pasted in an issue,
  shared in a screenshot), revoke and reissue it from the T-Invest API console
  immediately - do not just delete the exposing message/commit, since it may
  already be cached, indexed, or forked.

## Reporting a vulnerability

If you find a security issue in this repository (not in the T-Invest API
itself, which is out of scope), please report it privately via GitHub's
["Report a vulnerability"](../../security/advisories/new) flow on this repo
instead of opening a public issue, so it can be assessed before any details
are public.
