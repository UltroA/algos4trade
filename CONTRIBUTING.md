# Contributing

Thanks for considering a contribution. This project is a benchmark suite of
31+ algorithmic trading strategies backtested on real MOEX market data (see
`README.md` / `README_RU.md` for what it does and how to run it). Bug
reports, new algorithms, new `MarketDataProvider`/`Exchange` implementations,
and fixes to the benchmark scripts are all welcome.

For anything security-related (token handling, a real vulnerability, not a
correctness bug), see `SECURITY.md` instead of opening a public issue.

## Before you start

1. Read `README.md` section 1 (setup) and get `pytest`/the base benchmark
   scripts running locally - `pip install -e ".[dev]"` gets you both the
   dependencies and `pytest`.
2. For anything larger than a small fix, open an issue first describing what
   you want to change and why, so the approach can be discussed before you
   put time into an implementation.
3. If you're adding a new algorithm, data provider, or exchange, read the
   corresponding "How to add your own ..." section in the README first
   (sections 8-10) - each follows an existing, consistent OOP pattern
   (`BaseTradingAlgorithm`/`MarketDataProvider`/`Exchange`), and a PR that
   doesn't follow it will need rework.

## Code style

- Comments and docstrings are in English, formal register - no casual filler
  openers ("the idea is simple:", "basically"), no jokes, no first-person
  asides. Every sentence starts with a capital letter, including the
  paragraph right after a docstring's title line (a lowercase continuation
  right under a title reads as an informal aside, not a proper sentence).
- Comments explain the non-obvious **why** (a hidden constraint, the
  reasoning behind a workaround, evidence that pinned down a bug) - not what
  the code already says. Default to no comment when the code is
  self-explanatory.
- Identifiers (variables, functions, classes) are descriptive and
  professional - no placeholder/joke/pet names, no invented abbreviations
  beyond ones already established in the codebase (`rss`, `cpu`, `pnl`,
  `fit`/`predict`, etc.).
- Every algorithm subclasses `BaseTradingAlgorithm` via `SingleAssetAlgorithm`
  or `MultiAssetAlgorithm` (`src/core/base.py`). `generate_signals` must
  return a position in **[-1, 1]** using only information available at that
  point in time (no look-ahead), and `fit()` must only ever see the data
  it's explicitly passed.

## Tests

`tests/` (pytest, `pip install -e ".[dev]"`) parametrizes over every
algorithm class `core.algorithm_discovery.discover_algorithms()` finds, and
asserts each one fits and produces signals in `[-1, 1]` on synthetic data -
the same contract every algorithm's own `if __name__ == "__main__":` smoke
test already checks manually. Run it with:

```bash
pytest
```

CI (`.github/workflows/ci.yml`) runs the same suite on every push/PR. A new
algorithm needs no new test file - it's picked up automatically as long as
it's discoverable (see README section 8/9/10 and
`core/algorithm_discovery.py`'s instantiability rule).

## Commits and sign-off (DCO)

Commits should be signed off under the [Developer Certificate of Origin](https://developercertificate.org/)
to confirm you have the right to submit the contribution under this
project's license:

```bash
git commit -s -m "your message"
```

This adds a `Signed-off-by: Your Name <email>` trailer to the commit. PRs
with unsigned commits may be asked to amend before merge.

## AI-assisted code

AI is used in this project (to verify correctness of certain algorithm
implementations and for initial English translation of in-code comments -
the model used so far is Claude Sonnet 5). If you contribute code written
with AI assistance, please:

- cover it with tests (see "Tests" above);
- verify it yourself for correctness - no unverified model hallucinations;
- explicitly mark generated fragments in the source where practical;
- note the use of AI in the commit/PR description, specifying exactly which
  changes were AI-generated;
- state which AI model was used.

## Pull requests

- Keep PRs focused - one fix or one feature per PR is easier to review than
  a bundle of unrelated changes.
- Update `README.md` **and** `README_RU.md` together if your change affects
  anything either one documents (setup, an algorithm's behavior, a script's
  flags, project structure) - they're meant to stay in sync.
- Fill in the PR template's test plan honestly - "ran `pytest`" is fine when
  that's genuinely what you did; note explicitly if something wasn't tested
  (e.g. a change to the live market simulator that needs real market hours
  to fully exercise).
