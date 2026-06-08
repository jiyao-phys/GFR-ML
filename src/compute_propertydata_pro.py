import argparse
import os
import numpy as np
import pandas as pd

EPS = 1e-12


def read_inputs(args):
    if args.excel:
        sheets = pd.read_excel(args.excel, sheet_name=None)
        if "data" in sheets and "property" in sheets:
            return sheets["data"].copy(), sheets["property"].copy()
        names = list(sheets.keys())
        if len(names) >= 2:
            return sheets[names[0]].copy(), sheets[names[1]].copy()
        raise ValueError("workbook needs 'data' and 'property' sheets, or at least two sheets")

    if args.data_xlsx and args.property_xlsx:
        data_df = pd.read_excel(args.data_xlsx, sheet_name=0)
        prop_df = pd.read_excel(args.property_xlsx, sheet_name=0)
        return data_df, prop_df

    if os.path.exists("data.xlsx") and os.path.exists("property.xlsx"):
        data_df = pd.read_excel("data.xlsx", sheet_name=0)
        prop_df = pd.read_excel("property.xlsx", sheet_name=0)
        return data_df, prop_df

    raise ValueError(
        "provide --excel, or --data_xlsx and --property_xlsx, "
        "or place data.xlsx and property.xlsx in the working directory"
    )


def _amounts(row, cols):
    try:
        return row[cols].astype(float).values
    except (TypeError, ValueError):
        return pd.to_numeric(row[cols].replace("", np.nan), errors="coerce").fillna(0).values.astype(float)


def _norm_weights(proportions, present):
    idx = np.where(present)[0]
    if len(idx) == 0:
        return idx, np.array([])
    p = proportions[present]
    s = p.sum()
    if s <= EPS:
        return idx, np.ones(len(idx)) / len(idx)
    return idx, p / s


def _prop_stats(t_vals, p_vals, colbase):
    keys = (
        f"{colbase}_max", f"{colbase}_min", f"{colbase}_mean", f"{colbase}_wmean",
        f"{colbase}_wgeom", f"{colbase}_extreme_range", f"{colbase}_wrange",
        f"{colbase}_std", f"{colbase}_wstd",
    )
    nan_row = dict.fromkeys(keys, np.nan)
    if t_vals.size == 0:
        return nan_row

    valid = ~np.isnan(t_vals)
    if valid.sum() == 0:
        return nan_row

    t = t_vals[valid]
    p = p_vals[valid]
    p_sum = p.sum()

    t_max = np.max(t)
    t_min = np.min(t)
    wmean = np.sum(p * t) / p_sum if p_sum > EPS else np.nan

    pos = t > EPS
    if pos.any() and p_sum > EPS:
        tp, pp = t[pos], p[pos]
        ps = pp.sum()
        wgeom = float(np.exp(np.sum(pp / ps * np.log(tp)))) if ps > EPS else np.nan
    else:
        wgeom = np.nan

    if p_sum > EPS:
        pt = p * t
        wrange = (np.max(pt) - np.min(pt)) / p_sum
        wstd = float(np.sqrt(np.sum(p * (t - wmean) ** 2) / p_sum)) if not np.isnan(wmean) else np.nan
    else:
        wrange = np.nan
        wstd = np.nan

    std = np.std(t, ddof=0) if len(t) > 1 else 0.0

    return {
        f"{colbase}_max": t_max,
        f"{colbase}_min": t_min,
        f"{colbase}_mean": np.mean(t),
        f"{colbase}_wmean": wmean,
        f"{colbase}_wgeom": wgeom,
        f"{colbase}_extreme_range": t_max - t_min,
        f"{colbase}_wrange": wrange,
        f"{colbase}_std": std,
        f"{colbase}_wstd": wstd,
    }


def compute_propertydata(data_df, prop_df):
    label_col = data_df.columns[-1]
    comp_cols = list(data_df.columns[:-1])
    prop_idx = prop_df.set_index(prop_df.columns[0])
    prop_names = list(prop_df.columns[1:])

    rows = []
    for _, row in data_df.iterrows():
        amounts = _amounts(row, comp_cols)
        total = amounts.sum()
        props = amounts / total if total > EPS else np.zeros_like(amounts)

        present = amounts > EPS
        _, p_norm = _norm_weights(props, present)
        present_comps = [comp_cols[i] for i in np.where(present)[0]]

        out = {f"comp_prop_{c}": props[i] for i, c in enumerate(comp_cols)}

        for prop in prop_names:
            colbase = prop.replace(" ", "_")
            try:
                t_vals = prop_idx[prop].reindex(present_comps).astype(float).values
            except (TypeError, ValueError):
                t_vals = np.full(len(present_comps), np.nan)
            out.update(_prop_stats(t_vals, p_norm, colbase))

        out[label_col] = row[label_col]
        rows.append(out)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--excel")
    p.add_argument("--data_xlsx")
    p.add_argument("--property_xlsx")
    p.add_argument("--out", default="propertydata.xlsx")
    args = p.parse_args()

    data_df, prop_df = read_inputs(args)
    result = compute_propertydata(data_df, prop_df)
    result.to_excel(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
