# Agent instructions for this repo

## Read `references/` before grepping the codebase

This repo has a maintained, index-driven documentation tree at
`references/INDEX.md`. Before running `grep`/`find`/exploratory reads across
`kaggle/`, `scripts/`, or `src/` to understand *what a piece of the system
does or why it's built that way*, read `references/INDEX.md` first and follow
it to the specific domain sub-index (`dataset/`, `roi_cache/`,
`armc_roi_training/`, `web_annotator/`, `model_weights/`) and only the file(s)
your task needs.

Why: the reference tree already contains function/class inventories, data
contracts, architecture diagrams, and chronological incident history that
would otherwise take many grep/read round-trips to reconstruct, and doing so
burns tokens on content someone already wrote down deliberately.

How to apply:
- Use `references/` to answer "what does X do", "why is Y shaped this way",
  "what's the data contract for Z" — trust it for orientation.
- Do not treat it as ground truth for an exact line number, current function
  signature, or anything you're about to edit — verify against the real file
  first, since the docs can drift (they say so themselves; e.g. rerun the EDA
  script before trusting stale numbers).
- Never read `references/archive/` — it's explicitly superseded, kept only
  for audit history.
- If a task spans multiple domains (e.g. a crop-size change touching both
  `roi_cache/` and `armc_roi_training/`), read each relevant domain's
  `INDEX.md`, not just one.
