# AGENTS.md: Efficiency Rules (paste at the start of any agent session)

Goal: finish the task correctly with the **fewest tokens and tool calls**. Quality is never traded for brevity, but waste is.

## 1. Before Acting
- Restate the task in **one line**. If anything blocking is unclear, ask **one** short question; otherwise state your assumption and proceed.
- Plan in **max 5 bullets** before coding. No long essays.
- Check what already exists (files, functions, deps) **before** writing new code.

## 2. Reading & Context
- Read only the files/lines needed. Use search (`grep`, `rg`) before opening whole files.
- Never re-read a file you already read unless it changed.
- Never paste large logs, datasets, or outputs. Use `head`, `tail`, `wc`, or a 5-row sample.
- Summarize long context into a 5-line state note and drop the rest.

## 3. Writing Output
- Return **only what changed** (diffs, patched functions), not whole files, unless the file is under ~50 lines or I ask.
- No preamble, no recap, no "Here is what I did" repeating the code.
- Explanations: max 3 sentences unless I ask "explain".
- No unsolicited alternatives, tutorials, or extra features.

## 4. Coding
- Smallest working change first. No refactors, renames, or new abstractions unless asked.
- Reuse existing libraries. Do not add a dependency for something under 20 lines.
- Run the **narrowest** test or command that proves the change works (one test, not the full suite).
- On error: read the **last 20 lines** of the trace, fix the root cause, retry once. If it fails twice, stop and report the cause plus 2 options.

## 5. AI/ML Specific
- Develop on a **small sample** (100 to 1,000 rows, 1 epoch, small model) first. Scale up only after it works.
- Cache datasets, embeddings, tokenized data, and model outputs. Never recompute.
- Fix random seeds. Log metrics to a file, not to chat.
- Do not print training progress per step. Log every N steps or per epoch.
- Prefer a smaller or cheaper model for drafts, classification, and routing. Use the large model only for hard reasoning.
- Keep prompts short and templated. Do not resend unchanged system prompts or few-shot examples when caching is available.

## 6. Web Dev Specific
- Edit the specific component or route, not the whole app.
- Do not regenerate boilerplate (config, package.json, CSS resets) that already exists.
- Verify with a targeted check (build, lint a single file, hit one endpoint) instead of a full pipeline.
- Keep CSS and JS changes scoped. No design overhauls unless asked.

## 7. Stop Conditions
- Stop when the acceptance criteria are met. Do not keep polishing.
- End with: `Done: <1 line> | Files: <list> | Next (optional): <1 line>`

## 8. Budget Guardrails
| Task size | Max tool calls | Max output |
|---|---|---|
| Quick fix | 5 | ~150 words + diff |
| Feature | 15 | ~300 words + diffs |
| Large task | split into steps, confirm after each | |

If a task will exceed its budget, **say so and split it** instead of pushing through.
