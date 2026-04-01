import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

from preprocessing import segment_and_preprocess_sp


def extract_features(segments, subject_id):
    results = []

    for label, data in segments.items():
        raw_sig = data["raw"]
        sig = data["normalized"]
        time_arr = data["time"]

        if len(sig) < 10:
            continue

        # ========= 时域 =========
        diff1 = np.diff(sig)
        diff2 = np.diff(diff1)

        features = {
            "Subject_ID": subject_id,
            "Stage": label,

            "max": np.max(raw_sig),
            "min": np.min(raw_sig),
            "median": np.median(sig),
            "mean": np.mean(sig),
            "var": np.var(sig),
            "rms": np.sqrt(np.mean(sig**2)),

            "diff1_mean": np.mean(diff1) if len(diff1) else 0,
            "diff2_mean": np.mean(diff2) if len(diff2) else 0,
            "diff1_std": np.std(diff1) if len(diff1) else 0,
            "diff2_std": np.std(diff2) if len(diff2) else 0,
        }

        # ========= 频域 =========
        total_seconds = (time_arr[-1] - time_arr[0]).astype('timedelta64[s]').astype(float)

        if total_seconds <= 0:
            freq_feats = [0]*8
        else:
            t = np.linspace(0, total_seconds, len(sig))
            uniform_t = np.arange(0, total_seconds, 1.0)

            if len(uniform_t) >= 2:
                interp_func = interp1d(t, sig, kind='linear', fill_value="extrapolate")
                uniform_sig = interp_func(uniform_t)

                freqs, psd = signal.welch(
                    uniform_sig,
                    fs=1.0,
                    nperseg=min(256, len(uniform_sig))
                )

                bands = [(i/16, (i+1)/16) for i in range(8)]
                freq_feats = [
                    np.sum(psd[(freqs >= low) & (freqs < high)])
                    for low, high in bands
                ]
            else:
                freq_feats = [0]*8

        for i in range(8):
            features[f"freq{i}"] = freq_feats[i]

        results.append(features)

    return results