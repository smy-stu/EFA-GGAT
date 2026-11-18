import pandas as pd
import numpy as np
import re

input_file = "Original geochemical elements.xlsx"
output_file = "Processed_geochemical_elements.xlsx"
sheet_name = "Sheet1"

df = pd.read_excel(input_file, sheet_name=sheet_name)
df.columns = [str(c).strip() for c in df.columns]

# Element columns: from SiO2 to Lu
start = df.columns.get_loc("SiO2")
end = df.columns.get_loc("Lu")
element_cols = df.columns[start:end + 1].tolist()
meta_cols = [c for c in df.columns if c not in element_cols]

# Numeric parsing: handle <0.01, comma decimals such as 41,59 / 1,28, etc.
def parse_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan

    # Detection limit <0.01 is treated as 0.005
    if s.startswith("<"):
        try:
            return float(s[1:].replace("，", ",").replace(",", ".")) / 2.0
        except Exception:
            return np.nan

    if s.startswith(">"):
        try:
            return float(s[1:].replace("，", ",").replace(",", "."))
        except Exception:
            return np.nan

    s = s.replace("，", ",").replace(",", ".")
    s = re.sub(r"[^0-9eE+\-.]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

for c in element_cols:
    df[c] = df[c].map(parse_num)

# 1) Remove element columns with missing rate >70%
miss_rate = df[element_cols].isna().mean()
delete_cols = miss_rate[miss_rate > 0.70].index.tolist()
keep_element_cols = [c for c in element_cols if c not in delete_cols]

missing_report = pd.DataFrame({
    "column": miss_rate.index,
    "missing_rate": miss_rate.values
})
missing_report["delete_gt_70pct"] = missing_report["missing_rate"] > 0.70

# Optional: keep the missing-rate report as a separate CSV
missing_report.to_csv("missing_rate_report.csv", index=False, encoding="utf-8-sig")

# 2) After removing columns, median-impute the remaining element columns
df_imputed = df[meta_cols + keep_element_cols].copy()
for c in keep_element_cols:
    med = df_imputed[c].median()
    df_imputed[c] = df_imputed[c].fillna(med)

# Specify transformation columns
clr_cols = [
    "SiO2", "TiO2", "Al2O3", "TFe2O3",
    "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5"
]
log_cols = [
    "Li", "Be", "Sc", "V", "Cr", "Co", "Ni", "Cu", "Zn",
    "Ga", "Rb", "Ba", "Sr", "Y", "Zr", "Nb", "Hf", "Ta",
    "Pb", "Th", "U"
]
ree_cols = [
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
]

# Automatically skip any column removed due to missing rate >70%
clr_cols = [c for c in clr_cols if c in df_imputed.columns]
log_cols = [c for c in log_cols if c in df_imputed.columns]
ree_cols = [c for c in ree_cols if c in df_imputed.columns]

df_trans = df_imputed.copy()

# 3) clr transformation
# clr(x_i) = log(x_i) - mean(log(x))
# clr requires positive values; <=0 is replaced by half of the minimum positive value in that column
X = df_imputed[clr_cols].astype(float).copy()
for c in clr_cols:
    pos = X[c][X[c] > 0]
    if len(pos) == 0:
        raise ValueError(f"{c} has no positive values; cannot perform clr transformation")
    fill_value = pos.min() / 2.0
    X[c] = X[c].mask(X[c] <= 0, fill_value)

logX = np.log(X)
clr = logX.sub(logX.mean(axis=1), axis=0)

# Add clr_ columns, keep imputed original columns
for c in clr_cols:
    df_trans["clr_" + c] = clr[c]

# 4) log10 transformation
# <=0 is replaced by half of the minimum positive value in that column
for c in log_cols:
    s = df_imputed[c].astype(float).copy()
    pos = s[s > 0]
    if len(pos) == 0:
        raise ValueError(f"{c} has no positive values; cannot perform log10 transformation")
    fill_value = pos.min() / 2.0
    s = s.mask(s <= 0, fill_value)
    df_trans["log_" + c] = np.log10(s)

# 5) REE not transformed: La–Lu in df_trans remain median-imputed values from df_imputed

# 6) Output: only the second part, i.e. the Transformed sheet
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_trans.to_excel(writer, sheet_name="Transformed", index=False)

print("Deleted columns:", delete_cols)
print("Retained element columns:", keep_element_cols)
print("clr transformed columns:", clr_cols)
print("log10 transformed columns:", log_cols)
print("REE untransformed columns:", ree_cols)
print(f"Output written: {output_file}")
print("Excel output contains only the Transformed sheet.")