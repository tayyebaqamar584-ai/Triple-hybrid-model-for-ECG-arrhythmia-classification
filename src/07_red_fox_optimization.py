"""
Red Fox Optimization (RFO) for the Proposed Model's meta-learner hyperparameters.

Implements the exploration / exploitation / reproduction structure described in
Polap & Wozniak (2021), "Red fox optimization algorithm," Expert Systems with
Applications 166:114107. This is our own implementation of that structure (the
paper does not publish a reference implementation), applied here to select
XGBoost hyperparameters for the Layer III meta-learner.

Fitness is evaluated ONLY on an 80/20 stratified holdout carved out of the
VALIDATION split (same holdout used by the existing leakage-safe tuning script)
-- never on the test split -- so this stays leakage-safe throughout the search.
The final best configuration is refit on the full validation split and
evaluated once, at the end, on the untouched test split.
"""
import json
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

MET = "results_beatwise/metrics"

# ---------------- Data (Layer II probabilities -> Layer III meta-input) ----------------
y_val = np.load(f"{MET}/y_val.npy")
y_test = np.load(f"{MET}/y_test.npy")

dh1v2_val = np.load(f"{MET}/dh1v2_probs_val.npy")
dh2v2_val = np.load(f"{MET}/dh2v2_probs_val.npy")
dh3v2_val = np.load(f"{MET}/dh3v2_probs_val.npy")
dh1v2_test = np.load(f"{MET}/dh1v2_probs_test.npy")
dh2v2_test = np.load(f"{MET}/dh2v2_probs_test.npy")
dh3v2_test = np.load(f"{MET}/dh3v2_probs_test.npy")

meta_val = np.concatenate([dh1v2_val, dh2v2_val, dh3v2_val], axis=1)
meta_test = np.concatenate([dh1v2_test, dh2v2_test, dh3v2_test], axis=1)

# Same 80/20 stratified holdout split used by the existing (v2) tuning script
Xtr, Xho, ytr, yho = train_test_split(meta_val, y_val, test_size=0.2, stratify=y_val, random_state=RNG_SEED)

# ---------------- Search space ----------------
# [n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda]
BOUNDS_LOW = np.array([100, 3, 0.01, 0.6, 0.6, 1, 0.0, 0.5])
BOUNDS_HIGH = np.array([600, 8, 0.30, 1.0, 1.0, 10, 1.0, 2.0])
INT_DIMS = [0, 1, 5]  # n_estimators, max_depth, min_child_weight
D = 8

def clip_and_round(x):
    x = np.clip(x, BOUNDS_LOW, BOUNDS_HIGH)
    x = x.copy()
    for i in INT_DIMS:
        x[i] = round(x[i])
    return x




def vec_to_params(x):
    return dict(
        n_estimators=int(x[0]), max_depth=int(x[1]), learning_rate=float(x[2]),
        subsample=float(x[3]), colsample_bytree=float(x[4]), min_child_weight=int(x[5]),
        reg_alpha=float(x[6]), reg_lambda=float(x[7]),
    )



_fitness_cache = {}


def fitness(x):
    key = tuple(np.round(x, 5))
    if key in _fitness_cache:
        return _fitness_cache[key]
    params = vec_to_params(x)
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=5, eval_metric="mlogloss",
        random_state=RNG_SEED, n_jobs=-1, **params,
    )
    model.fit(Xtr, ytr)
    pred_ho = model.predict(Xho)
    f1 = f1_score(yho, pred_ho, average="macro")
    _fitness_cache[key] = f1
    return f1



# ---------------- Red Fox Optimization ----------------

POP_SIZE = 12
N_ITERS = 25
REPRO_FRACTION = 0.15  # worst fraction replaced by reproduction each iteration

t0 = time.time()
population = np.array([BOUNDS_LOW + rng.random(D) * (BOUNDS_HIGH - BOUNDS_LOW) for _ in range(POP_SIZE)])
population = np.array([clip_and_round(p) for p in population])
fit = np.array([fitness(p) for p in population])

best_idx = np.argmax(fit)
global_best = population[best_idx].copy()
global_best_fit = fit[best_idx]

history = [{"iteration": 0, "best_fitness": float(global_best_fit), "mean_fitness": float(fit.mean())}]
n_repro = max(1, int(round(REPRO_FRACTION * POP_SIZE)))

for t in range(1, N_ITERS + 1):
    alpha = 2.0 * (1.0 - t / N_ITERS)  # decreasing control parameter: exploration -> exploitation

    for i in range(POP_SIZE):
        if i == best_idx:
            continue
        r = rng.random()
        if r < 0.5:
            # Exploration: move toward the global best ("scent of prey"), scaled by alpha and random noise
            step = alpha * rng.random(D) * (global_best - population[i])
            candidate = population[i] + step
        else:
            # Exploitation: local random walk around current position, shrinking as alpha decreases
            local_scale = alpha * 0.15 * (BOUNDS_HIGH - BOUNDS_LOW)
            candidate = population[i] + (2 * rng.random(D) - 1) * local_scale
        candidate = clip_and_round(candidate)
        cand_fit = fitness(candidate)
        if cand_fit > fit[i]:  # greedy replacement
            population[i] = candidate
            fit[i] = cand_fit

    # Reproduction / habitat phase: replace worst individuals with offspring of the two best ("alpha pair")
    order = np.argsort(fit)
    worst_idxs = order[:n_repro]
    parent_a, parent_b = population[order[-1]], population[order[-2]]
    for wi in worst_idxs:
        cross = rng.random(D)
        child = cross * parent_a + (1 - cross) * parent_b
        noise_scale = 0.05 * (BOUNDS_HIGH - BOUNDS_LOW)
        child = child + rng.normal(0, 1, D) * noise_scale
        child = clip_and_round(child)
        child_fit = fitness(child)
        population[wi] = child
        fit[wi] = child_fit

    best_idx = int(np.argmax(fit))
    if fit[best_idx] > global_best_fit:
        global_best = population[best_idx].copy()
        global_best_fit = fit[best_idx]

    history.append({"iteration": t, "best_fitness": float(global_best_fit), "mean_fitness": float(fit.mean())})
    print(f"iter {t:3d}  best_holdout_f1={global_best_fit:.4f}  mean={fit.mean():.4f}  alpha={alpha:.3f}")

elapsed = time.time() - t0
print(f"\nRFO finished in {elapsed:.1f}s, {len(_fitness_cache)} unique fitness evaluations")
best_params = vec_to_params(global_best)
print("Best params (selected on VAL holdout only):", best_params)
print("Best holdout F1:", global_best_fit)

# ---------------- Final refit on FULL validation split, single evaluation on TEST ----------------
final_model = xgb.XGBClassifier(
    objective="multi:softprob", num_class=5, eval_metric="mlogloss",
    random_state=RNG_SEED, n_jobs=-1, **best_params,
)
final_model.fit(meta_val, y_val)
proba_test = final_model.predict_proba(meta_test)
pred_test = np.argmax(proba_test, axis=1)

test_acc = accuracy_score(y_test, pred_test)
test_f1_macro = f1_score(y_test, pred_test, average="macro")
test_f1_weighted = f1_score(y_test, pred_test, average="weighted")
test_kappa = cohen_kappa_score(y_test, pred_test)

print("\n=== FINAL RFO-TUNED PROPOSED MODEL (evaluated once on TEST) ===")
print("accuracy:", test_acc, "f1_macro:", test_f1_macro, "f1_weighted:", test_f1_weighted, "kappa:", test_kappa)

# ---------------- Save everything ----------------
np.save(f"{MET}/proposed_rfo_probs_test.npy", proba_test)

with open(f"{MET}/proposed_rfo_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

with open(f"{MET}/proposed_rfo_convergence_history.json", "w") as f:
    json.dump(history, f, indent=2)

with open(f"{MET}/proposed_rfo_test_metrics.json", "w") as f:
    json.dump({
        "model_name": "PROPOSED_RFO",
        "selection_method": "Red Fox Optimization, fitness = validation-holdout macro-F1 (leakage-safe)",
        "best_params": best_params,
        "holdout_f1_at_selection": float(global_best_fit),
        "accuracy": float(test_acc),
        "f1_macro": float(test_f1_macro),
        "f1_weighted": float(test_f1_weighted),
        "kappa": float(test_kappa),
        "n_test": int(len(y_test)),
        "elapsed_seconds": elapsed,
        "unique_fitness_evals": len(_fitness_cache),
    }, f, indent=2)

print("\nSaved: proposed_rfo_probs_test.npy, proposed_rfo_best_params.json, "
      "proposed_rfo_convergence_history.json, proposed_rfo_test_metrics.json")
