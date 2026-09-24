"""Stage 2: turn per-family scores into 8-class probabilities and a macro-F1-optimised decision."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from streams import FAM, LABELS
from protocol import macro_f1

def s2_features(S):
    X = np.log(np.clip(S[FAM].values, 1e-4, 1-1e-4) / (1 - np.clip(S[FAM].values, 1e-4, 1-1e-4)))
    srt = np.sort(X, axis=1)[:, ::-1]
    return np.hstack([X, srt[:, :2], (srt[:, 0] - srt[:, 1])[:, None]])

def fit_stage2(S, y, C=1.0, seed=0):
    lr = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    lr.fit(s2_features(S), pd.Series(y).map(LABELS.index).values)
    return lr

def predict_proba(lr, S):
    P = np.zeros((len(S), 8)); P[:, lr.classes_] = lr.predict_proba(s2_features(S))
    return pd.DataFrame(P, index=S.index, columns=LABELS)

def tune_weights(P, y, n_rounds=3, grid=np.linspace(0.5, 2.5, 21)):
    """coordinate ascent on per-class multiplicative weights to maximise macro-F1."""
    w = np.ones(8); y = np.asarray(y); Pv = P[LABELS].values
    best = macro_f1(y, np.array(LABELS)[(Pv * w).argmax(1)])
    for _ in range(n_rounds):
        for j in range(8):
            for g in grid:
                w2 = w.copy(); w2[j] = g
                f = macro_f1(y, np.array(LABELS)[(Pv * w2).argmax(1)])
                if f > best + 1e-9: best, w = f, w2
    return w, best

def decide_w(P, w):
    return np.array(LABELS)[(P[LABELS].values * w).argmax(1)]
