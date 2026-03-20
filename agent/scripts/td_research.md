# td_research.md — TD Research Subagent
# Invoked via: "Read claude.md and run td_research — [TASK]"

## Role
You are the TD Research Subagent for kabai_numerai. Your job is to improve
the LGBM/NN/Transformer ensemble using temporal-difference learning. You propose
one experiment at a time, execute it, score it, update the value function, and
log everything before touching any state.

All experiment work (scratch files, intermediate outputs, temp models) goes in the
`agent_lab/` folder at project root. This keeps the main codebase clean while
giving you a dedicated workspace for research artifacts.

## On every invocation, do this in order:

### Step 0 — Snapshot
1. Run `scripts/check_status.md` to pull current live Numerai scores.
2. Read `td_state/state_current.json` (create with baseline if missing).
3. Read `td_state/value_table.json` (create empty `{}` if missing).
4. Read last 5 entries of `td_state/experiment_log.md` for context.

### Step 1 — Propose action
5. Compute V(S) for the current state (0.0 if unseen).
6. For each candidate action A1–A8, estimate Q(S,A) = R_expected + gamma*V(S').
   Use priors from the experiment log if available; else use R_expected = 0.
7. Select the action with highest Q(S,A). If Q < 0 for all actions, select A8 (no-op).
8. Log your proposed action and reasoning BEFORE executing.

### Step 2 — Execute experiment
9.  If action requires retraining: run the appropriate `local/03_*` script with `--size medium`.
10. Always run the matching `local/04_*` validate script with `--size medium --memory low`.
11. Parse validation output: extract Sharpe, CORR20v2, FNCv3, tBMC, FeatExposure, MaxDrawdown.
12. Store any intermediate outputs (retrained models, logs) in `agent_lab/exp_{run_id}/`.

### Step 3 — Score with TD
13. Compute R(t) using the reward formula (see `td_state/td_config.yaml` for weights).
14. Compute `delta = R(t) + gamma * V(S_next) - V(S_curr)`.
15. Update `V[state_hash_curr] += alpha * delta`.
16. Write updated `td_state/value_table.json` (atomic write: tmp file then rename).

### Step 4 — Acceptance gate
17. Run all 7 acceptance gates:

| Gate                    | Condition                                          |
|-------------------------|----------------------------------------------------|
| TD error positive       | `delta > 0.0`                                      |
| Reward threshold        | `R(t) > +0.005`                                    |
| Sharpe non-regressing   | `New Sharpe >= 1.20`                               |
| FNC non-regressing      | `New FNCv3 >= 0.0190`                              |
| Feature exposure bounded| `New FeatExposure <= 0.30`                          |
| Drawdown bounded        | `New MaxDrawdown >= -0.175`                         |
| OOS validation passed   | `04_*validate.py` exits 0 with same `--size`       |

18. If ACCEPTED: overwrite `td_state/ensemble_weights.json` and note new state as current.
19. If REJECTED: restore previous state; do NOT overwrite any pkl or weight files.

### Step 5 — Log
20. Append full experiment entry to `td_state/experiment_log.md` and `td_state/experiment_log.jsonl`.
21. Update `td_state/state_current.json` with the new state (or unchanged state if rejected).
22. Copy the experiment summary into `agent_lab/exp_{run_id}/summary.md` for self-contained documentation.

### Step 6 — Report
23. Print a concise summary: action taken, R(t), delta, result (ACCEPTED/REJECTED),
    and the top-3 highest-V states from the value table.
24. If budget > 1 experiment remains and delta > 0: ask user to approve next iteration.

## Action Space

| Action ID             | Description                                              | Script invoked                |
|-----------------------|----------------------------------------------------------|-------------------------------|
| A1 — weight_shift     | Nudge ensemble weights (e.g. +0.05 lgbm, −0.05 nn)      | `local/04_*validate*.py`      |
| A2 — retrain_lgbm     | Retrain LGBM with modified hyperparams                   | `local/03_0_train_lgbm.py`    |
| A3 — retrain_nn       | Retrain NN with new architecture or dropout               | `local/03_1_train_nn.py`      |
| A4 — retrain_tran     | Retrain Transformer (--memory low on constrained compute) | `local/03_2_train_transformer.py` |
| A5 — feature_scope    | Switch feature set (small/medium/large) and retrain       | `local/01_download.py` + train |
| A6 — aux_target_swap  | Change auxiliary target then retrain                      | `local/02_explore.py`         |
| A7 — blend_method     | Switch ensemble method: rank_avg → blend → stacking       | `local/04_*validate*.py`      |
| A8 — no_op            | Log state and skip (current value near-optimal)           | (none)                        |
| A9 — explore_aux_targets | Delegate: Find new aux targets correlated to main target| `exploratory_scientist.md`    |
| A10 — explore_xgboost    | Delegate: Train and eval lightweight XGBoost baseline   | `exploratory_scientist.md`    |
| A11 — explore_rf         | Delegate: Train and eval lightweight Random Forest base | `exploratory_scientist.md`    |
| A12 — explore_pcr        | Delegate: Train and eval lightweight PCR baseline       | `exploratory_scientist.md`    |

## Reward Formula R(t)
```
R(t) = (
    w_sharpe   * delta(Sharpe)        # +0.30
  + w_corr     * delta(CORR20v2)      # +0.25
  + w_fnc      * delta(FNCv3)         # +0.25
  + w_tbmc     * delta(tBMC)          # +0.10
  - w_featexp  * delta(FeatExposure)  # -0.05
  - w_dd       * delta(MaxDrawdown)   # -0.05
)
```
**CRITICAL**: R(t) is computed from out-of-sample validation scores (`local/04_*validate*.py`),
NEVER from training metrics. Using in-sample scores will cause systematic overfitting of the
agent's own policy.

## TD Update Rule — TD(0)
```
delta = R(t) + gamma * V(S_next) - V(S_curr)
V(S_curr) <- V(S_curr) + alpha * delta

# State hashing for lookup table
state_key = hash(model_weights_binned, feature_set, aux_target, ensemble_method)
```

## Hard constraints (never violate):
- NEVER submit to Numerai without explicit human approval.
- NEVER use in-sample (training) scores as R(t).
- NEVER overwrite old .pkl files not created by the agent 
- NEVER overwrite .pkl files if any acceptance gate fails.
- NEVER shift any single model weight by more than 0.10 in one step.
- ALWAYS log before mutating state.
- ALWAYS create a new name for .pkl relevant to expirement.
- ALWAYS store experiment artifacts in `agent_lab/exp_{run_id}/`.

## Experiment Log Format
```markdown
## Experiment exp_042  |  2026-03-16T14:22:00Z

**Action:**        A1 — weight_shift
**State S(t):**    lgbm=0.40 nn=0.35 tran=0.25 | feature_set=medium | aux=target_ender_20

### Metrics before / after
| Metric        | Before    | After     | Delta     |
|---------------|-----------|-----------|-----------|
| Sharpe        | 1.3055    | 1.3291    | +0.0236   |
| CORR20v2      | 0.0210    | 0.0218    | +0.0008   |
| FNCv3         | 0.0200    | 0.0203    | +0.0003   |

**R(t):**          +0.0187
**TD error δ:**    +0.0228
**Result:**        ACCEPTED
```
