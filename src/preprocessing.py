import pandas as pd
import numpy as np
import pywt

STATE_LABELS = ['Rest', 'Positive', 'Neutral', 'Negative']


def segment_and_preprocess_sp(lines, df):

    # ========= 1. Extract timestamps =========
    if len(lines) < 2:
        raise ValueError("Invalid file format: missing timestamp line")

    start_times = [t for t in lines[1].strip().split(',') if t]

    if not start_times:
        raise ValueError("No timestamps extracted")

    # ========= 2. Clean dataframe =========
    df = df.copy()

    df['SP_Value'] = pd.to_numeric(df['SP_Value'], errors='coerce')
    df = df.dropna(subset=['Date', 'Time'])

    if len(df) == 0:
        raise ValueError("No valid data available")

    # ========= 3. Time alignment =========
    df['Date'] = df['Date'].astype(int).astype(str)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    base_date = df['Date'].iloc[0]
    event_times = pd.to_datetime([base_date + ' ' + t for t in start_times])

    segments = {}
    num_stages = min(len(STATE_LABELS), len(event_times))

    # ========= 4. Signal segmentation =========
    for i in range(num_stages):
        label = STATE_LABELS[i]
        start_t = event_times[i]

        if i < num_stages - 1:
            end_t = event_times[i + 1]
        else:
            end_t = start_t + pd.Timedelta(minutes=3.5)

        seg_df = df[(df['Datetime'] >= start_t) & (df['Datetime'] < end_t)].copy()
        sig = seg_df['SP_Value'].values.astype(float)

        if len(sig) == 0:
            continue

        # ========= 5. Outlier removal (IQR) =========
        Q1, Q3 = np.nanpercentile(sig, [25, 75])
        IQR = Q3 - Q1

        sig_clean = np.where(
            (sig < Q1 - 2.5 * IQR) | (sig > Q3 + 2.5 * IQR),
            np.nan,
            sig
        )

        # ========= 6. Interpolation =========
        nans = np.isnan(sig_clean)

        if np.any(nans) and not np.all(nans):
            x = lambda z: z.nonzero()[0]
            sig_clean[nans] = np.interp(x(nans), x(~nans), sig_clean[~nans])
        elif np.all(nans):
            sig_clean = np.zeros_like(sig_clean)

        # ========= 7. Wavelet denoising =========
        sig_denoised = sig_clean.copy()

        coeffs = pywt.wavedec(sig_denoised, 'db4', level=4)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        if sigma > 0:
            uthresh = sigma * np.sqrt(2 * np.log(len(sig_denoised)))
            coeffs = [coeffs[0]] + [
                pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]
            ]
            sig_denoised = pywt.waverec(coeffs, 'db4')[:len(sig_denoised)]

        # ========= 8. Normalization =========
        min_v, max_v = np.min(sig_denoised), np.max(sig_denoised)

        if max_v > min_v:
            sig_norm = (sig_denoised - min_v) / (max_v - min_v)
        else:
            sig_norm = np.zeros_like(sig_denoised)

        # ========= Store =========
        segments[label] = {
            "time": seg_df['Datetime'].values,
            "raw": sig_clean,
            "processed": sig_denoised,
            "normalized": sig_norm
        }

    return segments