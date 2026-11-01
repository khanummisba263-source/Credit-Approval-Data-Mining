# %% [markdown]
# # Credit Application Screening and Risk-Pattern Discovery (UCI Credit Approval)
# Part 1: set-up, exploratory data analysis (Module 1) and mixed-attribute similarity.

# %%
import os, time, json, sqlite3, warnings, math
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

DATA_PATH = "crx.data"
FIG, TAB, WH = "figures", "tables", "warehouse"
SEED = 42
for d in (FIG, TAB, WH):
    os.makedirs(d, exist_ok=True)

REPORT = {}   # tables for the Word report
NUM = {}      # scalar facts for the Word report


def fmt(v):
    if isinstance(v, (float, np.floating)):
        return "-" if np.isnan(v) else f"{v:.3f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def save_table(name, df, index=False):
    d = df.reset_index() if index else df
    d.to_csv(f"{TAB}/{name}.csv", index=False)
    REPORT[name] = {"columns": [str(c) for c in d.columns],
                    "rows": [[fmt(v) for v in r] for r in d.itertuples(index=False)]}


def savefig(name):
    plt.savefig(f"{FIG}/{name}.png")
    plt.close()


# %% [markdown]
# ## 1. Load the data
# crx.data has no header row and uses '?' for missing values (see crx.names).

# %%
cols = [f"A{i}" for i in range(1, 17)]
raw = pd.read_csv(DATA_PATH, header=None, names=cols, na_values="?")
y = (raw["A16"] == "+").astype(int).to_numpy()
X = raw.drop(columns="A16").copy()
NUM_COLS = ["A2", "A3", "A8", "A11", "A14", "A15"]
CAT_COLS = ["A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"]
NUM["n_rows"], NUM["n_attr"] = int(len(raw)), 15
NUM["n_pos"], NUM["n_neg"] = int(y.sum()), int((1 - y).sum())

# %% [markdown]
# ## 2. Data quality and attribute types

# %%
desc_map = {
    "A1": ("Nominal (2 values)", "b, a"), "A2": ("Continuous", "numeric"),
    "A3": ("Continuous", "numeric"), "A4": ("Nominal (4 values)", "u, y, l, t"),
    "A5": ("Nominal (3 values)", "g, p, gg"),
    "A6": ("Nominal (14 values)", "c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff"),
    "A7": ("Nominal (9 values)", "v, h, bb, j, n, z, dd, ff, o"),
    "A8": ("Continuous", "numeric"), "A9": ("Binary", "t, f"), "A10": ("Binary", "t, f"),
    "A11": ("Continuous (integer valued)", "numeric"), "A12": ("Binary", "t, f"),
    "A13": ("Nominal (3 values)", "g, p, s"), "A14": ("Continuous (integer valued)", "numeric"),
    "A15": ("Continuous (integer valued)", "numeric"), "A16": ("Class (binary)", "+, -"),
}
attr_rows = []
for c in cols:
    n_miss = int(raw[c].isna().sum())
    n_unique = int(raw[c].nunique())
    attr_rows.append([c, desc_map[c][0], desc_map[c][1], n_unique, n_miss,
                      round(100 * n_miss / len(raw), 2)])
attr_df = pd.DataFrame(attr_rows, columns=["Attribute", "Type", "Values / range",
                                           "Distinct values", "Missing", "Missing %"])
save_table("attribute_table", attr_df)

miss_cells = int(X.isna().sum().sum())
rows_with_miss = int(X.isna().any(axis=1).sum())
NUM["missing_cells"], NUM["rows_with_missing"] = miss_cells, rows_with_miss
NUM["rows_complete"] = int(len(X) - rows_with_miss)
miss_by_class = pd.crosstab(X.isna().any(axis=1).map({True: "has missing", False: "complete"}),
                            raw["A16"])
save_table("missing_by_class", miss_by_class, index=True)
chi_miss = stats.chi2_contingency(miss_by_class.values)
NUM["miss_class_p"] = float(chi_miss[1])

plt.figure(figsize=(6, 3))
m = X.isna().sum()
m = m[m > 0]
ax = sns.barplot(x=m.index, y=m.values, color="#4C72B0")
for i, v in enumerate(m.values):
    ax.text(i, v + 0.2, str(v), ha="center")
ax.set_ylabel("Missing values"); ax.set_xlabel("Attribute"); ax.set_title("Missing values per attribute")
savefig("01_missing_values")

# %% [markdown]
# ## 3. Summary statistics

# %%
summ = X[NUM_COLS].describe().T
summ["skew"] = X[NUM_COLS].skew()
summ["zeros %"] = [(X[c] == 0).mean() * 100 for c in NUM_COLS]
save_table("numeric_summary", summ.round(3), index=True)

cat_rows = []
for c in CAT_COLS:
    vc = X[c].value_counts()
    top = ", ".join([f"{k}:{v}" for k, v in vc.items()])
    cat_rows.append([c, top])
save_table("categorical_counts", pd.DataFrame(cat_rows, columns=["Attribute", "Value counts (observed)"]))

fig, axes = plt.subplots(2, 3, figsize=(8, 4.6))
for ax, c in zip(axes.ravel(), NUM_COLS):
    sns.histplot(data=X.assign(Class=raw["A16"]), x=c, hue="Class", bins=30, ax=ax,
                 palette={"+": "#55A868", "-": "#C44E52"}, legend=(c == "A2"))
    ax.set_title(c)
plt.tight_layout(); savefig("02_numeric_histograms")

fig, axes = plt.subplots(2, 3, figsize=(8, 4.6))
for ax, c in zip(axes.ravel(), NUM_COLS):
    sns.boxplot(data=X.assign(Class=raw["A16"]), x="Class", y=c, ax=ax,
                palette={"+": "#55A868", "-": "#C44E52"})
    ax.set_title(c)
plt.tight_layout(); savefig("03_numeric_boxplots")

# outliers by IQR rule
out_rows = []
for c in NUM_COLS:
    q1, q3 = X[c].quantile([.25, .75]); iqr = q3 - q1
    n_out = int(((X[c] < q1 - 1.5 * iqr) | (X[c] > q3 + 1.5 * iqr)).sum())
    out_rows.append([c, n_out, round(100 * n_out / len(X), 1)])
save_table("outliers_iqr", pd.DataFrame(out_rows, columns=["Attribute", "IQR outliers", "% of rows"]))

# class balance
plt.figure(figsize=(3.6, 2.8))
ax = sns.countplot(x=raw["A16"], order=["+", "-"], palette=["#55A868", "#C44E52"])
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
ax.set_xlabel("Class (A16)"); ax.set_title("Class distribution")
savefig("04_class_distribution")

# %% [markdown]
# ## 4. Relationship of each attribute with the class

# %%
def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.values.sum()
    return math.sqrt(chi2 / (n * (min(ct.shape) - 1)))

assoc = []
for c in CAT_COLS:
    ok = X[c].notna()
    ct = pd.crosstab(X.loc[ok, c], raw.loc[ok, "A16"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    assoc.append([c, "chi-square", chi2, p, cramers_v(X.loc[ok, c], raw.loc[ok, "A16"])])
for c in NUM_COLS:
    ok = X[c].notna()
    a = X.loc[ok & (y == 1), c]; b = X.loc[ok & (y == 0), c]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    r = stats.pointbiserialr(y[ok], X.loc[ok, c])[0]
    assoc.append([c, "Welch t-test", t, p, abs(r)])
assoc_df = pd.DataFrame(assoc, columns=["Attribute", "Test", "Statistic", "p-value", "Effect size"])
assoc_df["p-value"] = assoc_df["p-value"].map(lambda v: f"{v:.2e}")
save_table("attribute_class_association", assoc_df)

fig, axes = plt.subplots(3, 3, figsize=(8, 7))
for ax, c in zip(axes.ravel(), CAT_COLS):
    order = X[c].value_counts().index
    sns.countplot(data=raw, x=c, hue="A16", order=order, ax=ax,
                  palette={"+": "#55A868", "-": "#C44E52"}, legend=(c == "A1"))
    ax.set_title(c); ax.set_xlabel("")
plt.tight_layout(); savefig("05_categorical_by_class")

corr = X[NUM_COLS].assign(Class=y).corr()
plt.figure(figsize=(4.8, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
plt.title("Correlation (numeric attributes and class)")
savefig("06_correlation")
save_table("numeric_correlation", corr.round(3), index=True)

# %% [markdown]
# ## 5. Mixed-attribute similarity (Gower), compared with Euclidean on encoded data

# %%
def gower_matrix(df, num_cols, cat_cols):
    """Gower dissimilarity: range-normalised absolute difference for numeric attributes,
    simple mismatch (0/1) for nominal ones, missing pairs skipped and averaged over valid ones."""
    n = len(df)
    S = np.zeros((n, n)); W = np.zeros((n, n))
    for c in num_cols:
        v = df[c].to_numpy(float)
        rng = np.nanmax(v) - np.nanmin(v)
        ok = ~np.isnan(v)
        mask = ok[:, None] & ok[None, :]
        d = np.abs(np.nan_to_num(v)[:, None] - np.nan_to_num(v)[None, :]) / rng
        S += np.where(mask, d, 0); W += mask
    for c in cat_cols:
        codes = pd.Categorical(df[c]).codes
        ok = codes >= 0
        mask = ok[:, None] & ok[None, :]
        S += np.where(mask, (codes[:, None] != codes[None, :]).astype(float), 0); W += mask
    return S / W

G = gower_matrix(X, NUM_COLS, CAT_COLS)
np.save(f"{TAB}/gower_matrix.npy", G)
NUM["gower_mean"] = float(G[np.triu_indices_from(G, 1)].mean())

# worked example on three rows
ex_idx = [0, 1, 2]
ex = X.loc[ex_idx, ["A1", "A2", "A3", "A4", "A9", "A10", "A12", "A14", "A15"]].copy()
ex.insert(0, "Row", ["r1", "r2", "r3"])
save_table("gower_example_rows", ex)
pairs = [(0, 1), (0, 2), (1, 2)]
save_table("gower_example_distances", pd.DataFrame(
    [[f"r{a+1} - r{b+1}", G[a, b], 1 - G[a, b]] for a, b in pairs],
    columns=["Pair", "Gower dissimilarity", "Gower similarity"]))

# binary proximity example: simple matching vs Jaccard on A9, A10, A12 (t = 1)
bin_cols = ["A9", "A10", "A12"]
B = (X[bin_cols] == "t").astype(int).to_numpy()
def smc(a, b): return float((a == b).mean())
def jac(a, b):
    both = ((a == 1) & (b == 1)).sum(); either = ((a == 1) | (b == 1)).sum()
    return float(both / either) if either else 1.0
save_table("binary_similarity_example", pd.DataFrame(
    [[f"r{a+1} - r{b+1}", smc(B[a], B[b]), jac(B[a], B[b])] for a, b in pairs],
    columns=["Pair", "Simple matching", "Jaccard"]))

# heatmap of 60 rows sorted by class
rng = np.random.RandomState(SEED)
idx = np.sort(rng.choice(len(X), 60, replace=False))
idx = idx[np.argsort(y[idx], kind="stable")]
plt.figure(figsize=(5, 4.2))
sns.heatmap(1 - G[np.ix_(idx, idx)], cmap="viridis", xticklabels=False, yticklabels=False,
            cbar_kws={"label": "Gower similarity"})
plt.title("Similarity of 60 random applications (class - first, then class +)")
savefig("07_gower_heatmap")

# neighbour purity: how often are the 5 nearest neighbours of the same class?
def purity(D, k=5):
    D = D.copy(); np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    return float((y[nn] == y[:, None]).mean())

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import pairwise_distances

pre_desc = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), NUM_COLS),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(sparse_output=False))]), CAT_COLS)])
Z_desc = pre_desc.fit_transform(X)
Z_num_only = StandardScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(X[NUM_COLS]))
raw_num = SimpleImputer(strategy="mean").fit_transform(X[NUM_COLS])
D_euc_enc = pairwise_distances(Z_desc)
D_euc_num = pairwise_distances(Z_num_only)
D_euc_raw = pairwise_distances(raw_num)
sim_rows = [["Gower (mixed attributes, missing skipped)", purity(G)],
            ["Euclidean, one-hot + Z-score", purity(D_euc_enc)],
            ["Euclidean, numeric only, Z-score", purity(D_euc_num)],
            ["Euclidean, numeric only, raw (unscaled)", purity(D_euc_raw)]]
save_table("neighbour_purity", pd.DataFrame(sim_rows, columns=["Distance measure", "5-NN class agreement"]))
iu = np.triu_indices(len(X), 1)
sub = np.random.RandomState(0).choice(len(iu[0]), 30000, replace=False)
NUM["rho_gower_euc"] = float(stats.spearmanr(G[iu][sub], D_euc_enc[iu][sub])[0])
NUM["rho_gower_rawnum"] = float(stats.spearmanr(G[iu][sub], D_euc_raw[iu][sub])[0])
NUM["base_rate_agreement"] = float(y.mean() ** 2 + (1 - y.mean()) ** 2)
# redundancy check: A4 versus A5
a45 = pd.crosstab(X["A4"].fillna("missing"), X["A5"].fillna("missing"))
save_table("a4_a5_crosstab", a45, index=True)
NUM["a4_a5_perfect"] = bool(((a45 > 0).sum(axis=1) == 1).all() and ((a45 > 0).sum(axis=0) == 1).all())
a1011 = pd.crosstab(X["A10"], (X["A11"] > 0).map({True: "A11 > 0", False: "A11 = 0"}))
save_table("a10_a11_crosstab", a1011, index=True)
NUM["a10_a11_perfect"] = bool(((a1011 > 0).sum(axis=1) == 1).all() and ((a1011 > 0).sum(axis=0) == 1).all())
print("Part 1 done")
# %% [markdown]
# # Part 2: Preprocessing (Module 2)
# Imputation, encoding, min-max and Z-score scaling, attribute selection and PCA.

# %%
import re
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

MISS_STRATEGIES = ["drop", "mean_mode", "median_mode", "knn", "missing_cat"]


def make_pre(miss="mean_mode", enc="onehot", scale="zscore", rules=False):
    """Build the preprocessing ColumnTransformer.
    miss  : drop (rows removed beforehand) | mean_mode | median_mode | knn | missing_cat
    enc   : onehot | ordinal      (nominal attributes)
    scale : none | minmax | zscore (numeric attributes)
    rules : keep native attribute types (for the rule-based classifier)"""
    if miss == "mean_mode":
        num_imp = SimpleImputer(strategy="mean")
    elif miss == "median_mode":
        num_imp = SimpleImputer(strategy="median")
    elif miss == "knn":
        num_imp = KNNImputer(n_neighbors=5)
    elif miss == "missing_cat":
        num_imp = SimpleImputer(strategy="median", add_indicator=not rules)
    else:  # drop: no NaN remain, the imputer is inert
        num_imp = SimpleImputer(strategy="mean")
    if miss == "missing_cat":
        cat_imp = SimpleImputer(strategy="constant", fill_value="missing")
    else:
        cat_imp = SimpleImputer(strategy="most_frequent")
    if rules:
        ct = ColumnTransformer([("num", num_imp, NUM_COLS), ("cat", cat_imp, CAT_COLS)],
                               verbose_feature_names_out=False)
        ct.set_output(transform="pandas")
        return ct
    scaler = {"none": None, "minmax": MinMaxScaler(), "zscore": StandardScaler()}[scale]
    num_steps = ([("scale", scaler)] if scaler is not None else []) + [("imp", num_imp)]
    if enc == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer([("num", Pipeline(num_steps), NUM_COLS),
                              ("cat", Pipeline([("imp", cat_imp), ("enc", encoder)]), CAT_COLS)])


def attr_of(feature_name):
    return re.match(r"^(?:num__|cat__)?(A\d+)", feature_name).group(1)


# %% [markdown]
# ## 1. Effect of missing-value handling on the distributions

# %%
Xn = X[NUM_COLS]
sc = StandardScaler().fit(Xn)
knn_imp = sc.inverse_transform(KNNImputer(n_neighbors=5).fit_transform(sc.transform(Xn)))
knn_df = pd.DataFrame(knn_imp, columns=NUM_COLS)
mean_df = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(Xn), columns=NUM_COLS)
med_df = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Xn), columns=NUM_COLS)
rows = []
for c in ["A2", "A14"]:
    rows.append([c, "Observed values only", Xn[c].mean(), Xn[c].std(), Xn[c].median()])
    rows.append([c, "Mean imputation", mean_df[c].mean(), mean_df[c].std(), mean_df[c].median()])
    rows.append([c, "Median imputation", med_df[c].mean(), med_df[c].std(), med_df[c].median()])
    rows.append([c, "KNN imputation (k=5, Z-scored)", knn_df[c].mean(), knn_df[c].std(), knn_df[c].median()])
save_table("imputation_numeric_effect",
           pd.DataFrame(rows, columns=["Attribute", "Method", "Mean", "Std. dev.", "Median"]))

rows = []
for c in ["A1", "A4", "A6", "A7"]:
    vc = X[c].value_counts(normalize=True)
    mode = X[c].mode()[0]
    after = X[c].fillna(mode).value_counts(normalize=True)
    top = vc.index[0]
    rows.append([c, mode, int(X[c].isna().sum()), f"{100 * vc[top]:.1f}%", f"{100 * after[top]:.1f}%"])
save_table("imputation_categorical_effect",
           pd.DataFrame(rows, columns=["Attribute", "Mode used", "Missing", "Share of mode before", "Share of mode after"]))

# %% [markdown]
# ## 2. Normalisation: min-max and Z-score

# %%
mm = MinMaxScaler().fit_transform(mean_df)
zs = StandardScaler().fit_transform(mean_df)
rows = []
for j, c in enumerate(NUM_COLS):
    rows.append([c, f"{mean_df[c].min():.2f} to {mean_df[c].max():.2f}", f"{mean_df[c].std():.2f}",
                 f"{mm[:, j].min():.0f} to {mm[:, j].max():.0f}", f"{mm[:, j].std():.3f}",
                 f"{zs[:, j].min():.2f} to {zs[:, j].max():.2f}", f"{zs[:, j].std():.2f}"])
save_table("scaling_comparison", pd.DataFrame(rows, columns=[
    "Attribute", "Raw range", "Raw std", "Min-max range", "Min-max std", "Z-score range", "Z-score std"]))

fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
for ax, (title, data) in zip(axes, [("Raw", mean_df.values), ("Min-max", mm), ("Z-score", zs)]):
    sns.boxplot(data=pd.DataFrame(data, columns=NUM_COLS), ax=ax, orient="h", fliersize=1.5)
    ax.set_title(title)
    if title == "Raw":
        ax.set_xscale("symlog")
plt.tight_layout(); savefig("08_scaling_boxplots")

# %% [markdown]
# ## 3. Attribute selection (three independent criteria and a consensus)

# %%
Xm = X.copy()
for c in NUM_COLS:
    Xm[c] = Xm[c].fillna(Xm[c].mean())
for c in CAT_COLS:
    Xm[c] = Xm[c].fillna(Xm[c].mode()[0])
Xord = Xm.copy()
for c in CAT_COLS:
    Xord[c] = pd.Categorical(Xord[c]).codes
discrete = np.array([c in CAT_COLS for c in X.columns])
mi = mutual_info_classif(Xord.values.astype(float), y, discrete_features=discrete, random_state=SEED)
mi_s = pd.Series(mi, index=X.columns)

# (b) statistical significance of association with the class (-log10 p)
sig = {}
for _, r in assoc_df.iterrows():
    sig[r["Attribute"]] = -math.log10(max(float(r["p-value"]), 1e-300))
sig_s = pd.Series(sig)[X.columns]

# (c) random-forest impurity importance aggregated over one-hot columns
pre_rf = make_pre("mean_mode", "onehot", "none")
Zrf = pre_rf.fit_transform(X)
names_rf = pre_rf.get_feature_names_out()
rf = RandomForestClassifier(n_estimators=500, random_state=SEED, min_samples_leaf=3).fit(Zrf, y)
imp = pd.Series(rf.feature_importances_, index=[attr_of(n) for n in names_rf]).groupby(level=0).sum()[X.columns]

sel = pd.DataFrame({"Mutual information": mi_s, "-log10 p-value": sig_s, "Random-forest importance": imp})
ranks = sel.rank(ascending=False)
sel["Mean rank"] = ranks.mean(axis=1)
sel = sel.sort_values("Mean rank")
sel.insert(0, "Consensus position", range(1, len(sel) + 1))
save_table("attribute_selection", sel.round(3), index=True)
SELECTED_TOP = list(sel.index[:6])
NUM["top_attrs"] = SELECTED_TOP

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
sns.barplot(x=mi_s.sort_values(ascending=False).values, y=mi_s.sort_values(ascending=False).index,
            ax=axes[0], color="#4C72B0"); axes[0].set_title("Mutual information with class")
sns.barplot(x=imp.sort_values(ascending=False).values, y=imp.sort_values(ascending=False).index,
            ax=axes[1], color="#DD8452"); axes[1].set_title("Random-forest importance")
plt.tight_layout(); savefig("09_attribute_selection")

# %% [markdown]
# ## 4. Principal component analysis

# %%
pre_pca = make_pre("mean_mode", "onehot", "zscore")
Zp = pre_pca.fit_transform(X)
names_pca = pre_pca.get_feature_names_out()
pca = PCA(random_state=SEED).fit(Zp)
cum = np.cumsum(pca.explained_variance_ratio_)
NUM["n_encoded_features"] = int(Zp.shape[1])
NUM["pca_90"], NUM["pca_95"] = int(np.argmax(cum >= .90) + 1), int(np.argmax(cum >= .95) + 1)
NUM["pca_pc1"], NUM["pca_pc2"] = float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])
save_table("pca_variance", pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(10)],
    "Variance explained": pca.explained_variance_ratio_[:10],
    "Cumulative": cum[:10]}))

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].bar(range(1, 21), pca.explained_variance_ratio_[:20], color="#4C72B0")
axes[0].plot(range(1, 21), cum[:20], "o-", color="#C44E52", ms=3)
axes[0].axhline(.9, ls="--", c="grey"); axes[0].axhline(.95, ls=":", c="grey")
axes[0].set_xlabel("Component"); axes[0].set_ylabel("Variance ratio"); axes[0].set_title("Scree plot and cumulative variance")
P = pca.transform(Zp)[:, :2]
axes[1].scatter(P[y == 0, 0], P[y == 0, 1], s=8, c="#C44E52", label="Class -", alpha=.6)
axes[1].scatter(P[y == 1, 0], P[y == 1, 1], s=8, c="#55A868", label="Class +", alpha=.6)
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].legend(); axes[1].set_title("First two components")
plt.tight_layout(); savefig("10_pca")

load = pd.DataFrame(pca.components_[:2].T, index=names_pca, columns=["PC1", "PC2"])
top_load = pd.concat([load["PC1"].abs().sort_values(ascending=False).head(6).index.to_series(),
                      load["PC2"].abs().sort_values(ascending=False).head(6).index.to_series()])
tl = pd.DataFrame({"PC1 top loadings": load["PC1"].abs().sort_values(ascending=False).head(6).index.str.replace(r"^(num|cat)__", "", regex=True),
                   "PC1 weight": load.loc[load["PC1"].abs().sort_values(ascending=False).head(6).index, "PC1"].values,
                   "PC2 top loadings": load["PC2"].abs().sort_values(ascending=False).head(6).index.str.replace(r"^(num|cat)__", "", regex=True),
                   "PC2 weight": load.loc[load["PC2"].abs().sort_values(ascending=False).head(6).index, "PC2"].values})
save_table("pca_loadings", tl.round(3))
print("Part 2 done")
# %% [markdown]
# # Part 3: Classification (Module 3a)
# Decision tree, rule-based (sequential covering) and KNN classifiers, with leakage-free
# repeated stratified 10-fold cross-validation.

# %%
import pickle
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve)
from functools import partial

N_JOBS = 1


def mi_mixed(Xf, yf):
    """Mutual information on the encoded matrix: first len(NUM_COLS) columns are continuous,
    the remaining (one-hot) columns are discrete."""
    mask = np.arange(Xf.shape[1]) >= len(NUM_COLS)
    return mutual_info_classif(Xf, yf, discrete_features=mask, random_state=SEED)

CV = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=SEED)
SCORING = {"acc": "accuracy", "prec": "precision", "rec": "recall", "f1": "f1", "auc": "roc_auc"}


# %% [markdown]
# ## 1. Rule-based classifier: sequential covering with FOIL gain
# Learns an ordered list of IF-THEN rules for the positive class (as in RIPPER/CN2).
# Each rule is grown one condition at a time using FOIL information gain, accepted only if its
# precision on the training data is high enough, then the covered rows are removed.

# %%
class SequentialCoveringRules(ClassifierMixin, BaseEstimator):
    def __init__(self, min_precision=0.85, min_cover=8, max_conds=4, n_thresholds=8, max_rules=15):
        self.min_precision = min_precision
        self.min_cover = min_cover
        self.max_conds = max_conds
        self.n_thresholds = n_thresholds
        self.max_rules = max_rules

    # --- candidate conditions -------------------------------------------------
    def _make_conditions(self, X):
        conds = []
        for c in X.columns:
            if c in self.num_cols_:
                v = X[c].astype(float).to_numpy()
                qs = np.unique(np.quantile(v, np.linspace(.1, .9, self.n_thresholds)))
                for t in qs:
                    conds.append((c, "<=", float(t)))
                    conds.append((c, ">", float(t)))
            else:
                for val in X[c].dropna().unique():
                    conds.append((c, "==", val))
        return conds

    @staticmethod
    def _eval(cond, X):
        c, op, v = cond
        if op == "<=":
            return (X[c].astype(float) <= v).to_numpy()
        if op == ">":
            return (X[c].astype(float) > v).to_numpy()
        return (X[c] == v).to_numpy()

    @staticmethod
    def cond_text(cond):
        c, op, v = cond
        return f"{c} {op} {v:.2f}" if op in ("<=", ">") else f"{c} = {v}"

    # --- learning ----------------------------------------------------------------
    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.asarray(y).astype(int)
        self.num_cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cond_ = self._make_conditions(X)
        M = np.column_stack([self._eval(c, X) for c in self.cond_])
        pos, neg = (y == 1), (y == 0)
        remaining = np.ones(len(y), bool)
        self.rules_ = []
        for _ in range(self.max_rules):
            if (remaining & pos).sum() < self.min_cover:
                break
            cov = remaining.copy()
            chosen = []
            for _ in range(self.max_conds):
                p0, n0 = (cov & pos).sum(), (cov & neg).sum()
                if n0 == 0 or p0 == 0:
                    break
                cc = cov[:, None] & M
                p1 = (cc & pos[:, None]).sum(0).astype(float)
                n1 = (cc & neg[:, None]).sum(0).astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    gain = p1 * (np.log2(p1 / (p1 + n1)) - np.log2(p0 / (p0 + n0)))
                gain[~np.isfinite(gain)] = -np.inf
                gain[p1 < self.min_cover] = -np.inf
                gain[chosen] = -np.inf
                j = int(np.argmax(gain))
                if not np.isfinite(gain[j]) or gain[j] <= 0:
                    break
                chosen.append(j)
                cov = cov & M[:, j]
            p, n = int((cov & pos).sum()), int((cov & neg).sum())
            if not chosen or p < self.min_cover or p / max(p + n, 1) < self.min_precision:
                break
            self.rules_.append(([self.cond_[j] for j in chosen], p, n))
            remaining &= ~cov
        self.default_prob_ = ((remaining & pos).sum() + 1) / (remaining.sum() + 2)
        self.classes_ = np.array([0, 1])
        return self

    # --- prediction ---------------------------------------------------------------
    def predict_proba(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        prob = np.full(len(X), np.nan)
        matched = np.zeros(len(X), bool)
        for conds, p, n in self.rules_:
            m = np.ones(len(X), bool)
            for cnd in conds:
                m &= self._eval(cnd, X)
            new = m & ~matched
            prob[new] = (p + 1) / (p + n + 2)
            matched |= m
        prob[~matched] = self.default_prob_
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @staticmethod
    def simplify(conds):
        """Merge several bounds on the same numeric attribute into one readable condition."""
        lo, hi, parts = {}, {}, []
        for c, op, v in conds:
            if op == ">":
                lo[c] = max(lo.get(c, -np.inf), v)
            elif op == "<=":
                hi[c] = min(hi.get(c, np.inf), v)
            else:
                parts.append(f"{c} = {v}")
        for c in sorted(set(lo) | set(hi)):
            if c in lo and c in hi:
                parts.append(f"{lo[c]:.2f} < {c} <= {hi[c]:.2f}")
            elif c in lo:
                parts.append(f"{c} > {lo[c]:.2f}")
            else:
                parts.append(f"{c} <= {hi[c]:.2f}")
        return parts

    def describe(self):
        lines = []
        for k, (conds, p, n) in enumerate(self.rules_, 1):
            lines.append((f"R{k}", " AND ".join(self.simplify(conds)), p, n, p / (p + n)))
        return lines


# %% [markdown]
# ## 2. Pipelines and experiment runner

# %%
def build_model(name, miss, enc, scale, reduction="none", k_feat=None):
    if name == "DT":
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=10, random_state=SEED)
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=7)
    else:
        clf = SequentialCoveringRules()
    if name == "RULES":
        return Pipeline([("pre", make_pre(miss, rules=True)), ("clf", clf)])
    steps = [("pre", make_pre(miss, enc, scale))]
    if reduction == "pca90":
        steps.append(("pca", PCA(n_components=0.90, random_state=SEED)))
    elif reduction == "pca95":
        steps.append(("pca", PCA(n_components=0.95, random_state=SEED)))
    elif reduction == "selectk":
        steps.append(("sel", SelectKBest(mi_mixed, k=k_feat)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def run_cv(name, miss, enc, scale, reduction="none", k_feat=None):
    Xs, ys = X, y
    if miss == "drop":
        keep = ~X.isna().any(axis=1).to_numpy()
        Xs, ys = X[keep], y[keep]
    t0 = time.time()
    res = cross_validate(build_model(name, miss, enc, scale, reduction, k_feat), Xs, ys, cv=CV,
                         scoring=SCORING, n_jobs=N_JOBS)
    row = dict(model=name, missing=miss, encoding=enc, scaling=scale, reduction=reduction, n_rows=len(ys),
               k_feat=k_feat if k_feat else 0, time_s=time.time() - t0)
    for m in SCORING:
        row[f"{m}_mean"] = res[f"test_{m}"].mean()
        row[f"{m}_std"] = res[f"test_{m}"].std()
    return row, res["test_acc"]


CACHE = f"{TAB}/_cache_experiments.pkl"
if os.path.exists(CACHE):
    exp_df, FOLDS = pickle.load(open(CACHE, "rb"))
else:
    rows, FOLDS = [], {}
    for miss in MISS_STRATEGIES:
        r, f = run_cv("RULES", miss, "native", "none")
        rows.append(r); FOLDS[("RULES", miss, "native", "none", "none", 0)] = f
        for model in ("DT", "KNN"):
            for enc in ("onehot", "ordinal"):
                for scale in ("none", "minmax", "zscore"):
                    r, f = run_cv(model, miss, enc, scale)
                    rows.append(r); FOLDS[(model, miss, enc, scale, "none", 0)] = f
    # dimensionality reduction (mean/mode imputation, one-hot, Z-score)
    for model in ("DT", "KNN"):
        for red, kf in (("pca90", None), ("pca95", None), ("selectk", 5), ("selectk", 10), ("selectk", 15)):
            r, f = run_cv(model, "mean_mode", "onehot", "zscore", red, kf)
            rows.append(r); FOLDS[(model, "mean_mode", "onehot", "zscore", red, kf or 0)] = f
    exp_df = pd.DataFrame(rows)
    pickle.dump((exp_df, FOLDS), open(CACHE, "wb"))
exp_df.round(4).to_csv(f"{TAB}/all_experiments.csv", index=False)


def pick(model, miss="mean_mode", enc="onehot", scale="zscore", reduction="none", k_feat=0):
    q = exp_df[(exp_df.model == model) & (exp_df.missing == miss) & (exp_df.encoding == enc) &
               (exp_df.scaling == scale) & (exp_df.reduction == reduction) & (exp_df.k_feat == k_feat)]
    return q.iloc[0]


def ms(r, m="acc"):
    return f"{100 * r[m + '_mean']:.1f} ± {100 * r[m + '_std']:.1f}"


# %% [markdown]
# ## 3. Summary tables

# %%
# 3.1 baseline comparison of the three classifiers
base = [("Decision tree (depth 4)", pick("DT")), ("Rule-based (sequential covering)", pick("RULES", enc="native", scale="none")),
        ("KNN (k = 7)", pick("KNN"))]
save_table("clf_baseline", pd.DataFrame([[n, ms(r), ms(r, "prec"), ms(r, "rec"), ms(r, "f1"), f"{r.auc_mean:.3f}"]
                                         for n, r in base],
                                        columns=["Classifier", "Accuracy %", "Precision %", "Recall %", "F1 %", "AUC"]))

# 3.2 effect of scaling
rows = []
for model, nm in (("DT", "Decision tree"), ("KNN", "KNN")):
    for sc_ in ("none", "minmax", "zscore"):
        r = pick(model, scale=sc_)
        rows.append([nm, {"none": "None (raw)", "minmax": "Min-max", "zscore": "Z-score"}[sc_], ms(r), ms(r, "f1"), f"{r.auc_mean:.3f}"])
r = pick("RULES", enc="native", scale="none")
rows.append(["Rule-based", "Not applicable (threshold rules)", ms(r), ms(r, "f1"), f"{r.auc_mean:.3f}"])
save_table("clf_scaling", pd.DataFrame(rows, columns=["Classifier", "Scaling", "Accuracy %", "F1 %", "AUC"]))

# 3.3 effect of missing-value handling (one-hot, Z-score)
rows = []
labels = {"drop": "Delete rows with missing", "mean_mode": "Mean / mode", "median_mode": "Median / mode",
          "knn": "KNN (k=5) / mode", "missing_cat": "Median + 'missing' category"}
for miss in MISS_STRATEGIES:
    d, k, rl = pick("DT", miss=miss), pick("KNN", miss=miss), pick("RULES", miss=miss, enc="native", scale="none")
    rows.append([labels[miss], int(d.n_rows), ms(d), ms(rl), ms(k)])
save_table("clf_missing", pd.DataFrame(rows, columns=["Missing-value handling", "Rows used", "Decision tree acc %", "Rule-based acc %", "KNN acc %"]))

# 3.4 effect of encoding (mean/mode)
rows = []
for model, nm in (("DT", "Decision tree"), ("KNN", "KNN")):
    for enc in ("onehot", "ordinal"):
        for sc_ in ("none", "zscore"):
            r = pick(model, enc=enc, scale=sc_)
            rows.append([nm, "One-hot" if enc == "onehot" else "Ordinal (label)", "Raw" if sc_ == "none" else "Z-score", ms(r), f"{r.auc_mean:.3f}"])
save_table("clf_encoding", pd.DataFrame(rows, columns=["Classifier", "Encoding", "Numeric scaling", "Accuracy %", "AUC"]))

# 3.5 dimensionality reduction
rows = []
for model, nm in (("DT", "Decision tree"), ("KNN", "KNN")):
    rows.append([nm, "All encoded features", int(NUM["n_encoded_features"]), ms(pick(model)), f"{pick(model).auc_mean:.3f}"])
    for red, lab in (("pca90", "PCA (90% variance)"), ("pca95", "PCA (95% variance)")):
        r = pick(model, reduction=red)
        rows.append([nm, lab, NUM["pca_90"] if red == "pca90" else NUM["pca_95"], ms(r), f"{r.auc_mean:.3f}"])
    for kf in (5, 10, 15):
        r = pick(model, reduction="selectk", k_feat=kf)
        rows.append([nm, f"Top-{kf} features (mutual information)", kf, ms(r), f"{r.auc_mean:.3f}"])
save_table("clf_reduction", pd.DataFrame(rows, columns=["Classifier", "Representation", "Features", "Accuracy %", "AUC"]))

# 3.6 corrected resampled t-test (Nadeau & Bengio) for key comparisons
def nb_test(a, b):
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / math.sqrt((1 / J + (1 / 9)) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))

K = lambda *a, **k: FOLDS[a]
comps = [
    ("KNN: Z-score vs raw", FOLDS[("KNN", "mean_mode", "onehot", "zscore", "none", 0)], FOLDS[("KNN", "mean_mode", "onehot", "none", "none", 0)]),
    ("KNN: min-max vs Z-score", FOLDS[("KNN", "mean_mode", "onehot", "minmax", "none", 0)], FOLDS[("KNN", "mean_mode", "onehot", "zscore", "none", 0)]),
    ("KNN: one-hot vs ordinal (Z-score)", FOLDS[("KNN", "mean_mode", "onehot", "zscore", "none", 0)], FOLDS[("KNN", "mean_mode", "ordinal", "zscore", "none", 0)]),
    ("Decision tree: Z-score vs raw", FOLDS[("DT", "mean_mode", "onehot", "zscore", "none", 0)], FOLDS[("DT", "mean_mode", "onehot", "none", "none", 0)]),
    ("Decision tree vs rule-based", FOLDS[("DT", "mean_mode", "onehot", "zscore", "none", 0)], FOLDS[("RULES", "mean_mode", "native", "none", "none", 0)]),
    ("Decision tree vs KNN (Z-score)", FOLDS[("DT", "mean_mode", "onehot", "zscore", "none", 0)], FOLDS[("KNN", "mean_mode", "onehot", "zscore", "none", 0)]),
    ("Rule-based vs KNN (Z-score)", FOLDS[("RULES", "mean_mode", "native", "none", "none", 0)], FOLDS[("KNN", "mean_mode", "onehot", "zscore", "none", 0)]),
]
rows = []
for nm, a, b in comps:
    t, p = nb_test(a, b)
    rows.append([nm, f"{100 * np.mean(a):.1f}", f"{100 * np.mean(b):.1f}", f"{100 * (np.mean(a) - np.mean(b)):+.1f}", f"{t:.2f}", f"{p:.3f}"])
save_table("clf_significance", pd.DataFrame(rows, columns=["Comparison (A vs B)", "Acc. A %", "Acc. B %", "Difference", "t", "p-value"]))

# %% [markdown]
# ## 4. Sensitivity of hyper-parameters (descriptive)

# %%
cv_s = StratifiedKFold(10, shuffle=True, random_state=SEED)
ks = list(range(1, 26, 2))
curves = {}
for sc_ in ("none", "minmax", "zscore"):
    curves[sc_] = [cross_val_predict(Pipeline([("pre", make_pre("mean_mode", "onehot", sc_)),
                                               ("clf", KNeighborsClassifier(k))]), X, y, cv=cv_s) for k in ks]
    curves[sc_] = [accuracy_score(y, p) for p in curves[sc_]]
plt.figure(figsize=(5, 3.2))
for sc_, lab in (("none", "Raw"), ("minmax", "Min-max"), ("zscore", "Z-score")):
    plt.plot(ks, curves[sc_], "o-", ms=3, label=lab)
plt.xlabel("k (number of neighbours)"); plt.ylabel("CV accuracy"); plt.legend(); plt.title("KNN: effect of k and scaling")
savefig("11_knn_k_scaling")
NUM["knn_best_k_z"] = int(ks[int(np.argmax(curves["zscore"]))]); NUM["knn_best_acc_z"] = float(max(curves["zscore"]))
NUM["knn_best_k_raw"] = int(ks[int(np.argmax(curves["none"]))]); NUM["knn_best_acc_raw"] = float(max(curves["none"]))

depths = list(range(1, 13))
tr, te = [], []
for d in depths:
    m = Pipeline([("pre", make_pre("mean_mode", "onehot", "none")),
                  ("clf", DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=SEED))])
    te.append(accuracy_score(y, cross_val_predict(m, X, y, cv=cv_s)))
    tr.append(accuracy_score(y, m.fit(X, y).predict(X)))
plt.figure(figsize=(5, 3.2))
plt.plot(depths, tr, "o-", ms=3, label="Training accuracy"); plt.plot(depths, te, "o-", ms=3, label="10-fold CV accuracy")
plt.xlabel("Maximum tree depth"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Decision tree: depth versus over-fitting")
savefig("12_tree_depth")
pr1 = cross_val_predict(Pipeline([("pre", make_pre("mean_mode", "onehot", "none")),
                                  ("clf", DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=SEED))]),
                        X, y, cv=cv_s, method="predict_proba")[:, 1]
maj = max(y.mean(), 1 - y.mean())
save_table("baseline_reference", pd.DataFrame([
    ["Majority class (always predict -)", f"{100 * maj:.1f}", "0.0", "-", "0.500"],
    ["Single split (depth-1 tree)", f"{100 * accuracy_score(y, pr1 >= .5):.1f}", f"{100 * recall_score(y, pr1 >= .5):.1f}",
     f"{100 * precision_score(y, pr1 >= .5):.1f}", f"{roc_auc_score(y, pr1):.3f}"]],
    columns=["Reference model", "CV accuracy %", "Recall %", "Precision %", "AUC"]))
NUM["majority_acc"] = float(maj); NUM["depth1_acc"] = float(accuracy_score(y, pr1 >= .5))
NUM["tree_depth_best"] = int(depths[int(np.argmax(te))]); NUM["tree_depth_best_acc"] = float(max(te))
NUM["tree_depth_full_train"] = float(tr[-1]); NUM["tree_depth_full_cv"] = float(te[-1])

mp = [0.7, 0.8, 0.85, 0.9, 0.95]
rows = []
for p_ in mp:
    mdl = Pipeline([("pre", make_pre("mean_mode", rules=True)), ("clf", SequentialCoveringRules(min_precision=p_))])
    pr = cross_val_predict(mdl, X, y, cv=cv_s)
    nrules = len(clone(mdl).fit(X, y).named_steps["clf"].rules_)
    rows.append([p_, nrules, 100 * accuracy_score(y, pr), 100 * recall_score(y, pr), 100 * precision_score(y, pr)])
sw = pd.DataFrame(rows, columns=["Min. rule precision", "Rules learned", "CV accuracy %", "Recall %", "Precision %"])
sw[["CV accuracy %", "Recall %", "Precision %"]] = sw[["CV accuracy %", "Recall %", "Precision %"]].round(1)
save_table("rules_precision_sweep", sw)

# %% [markdown]
# ## 5. Interpretable final models, error analysis and audit of the screening decisions

# %%
# 5.1 decision tree on all data (mean/mode, one-hot) and its rules
pre_f = make_pre("mean_mode", "onehot", "none")
Zf = pre_f.fit_transform(X)
fnames = [n.split("__", 1)[1] for n in pre_f.get_feature_names_out()]
dt = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=10, random_state=SEED).fit(Zf, y)
txt = export_text(dt, feature_names=fnames, show_weights=True)
open(f"{TAB}/decision_tree_rules.txt", "w").write(txt)
NUM["dt_train_acc"] = float(dt.score(Zf, y)); NUM["dt_leaves"] = int(dt.get_n_leaves())
plt.figure(figsize=(16, 6.4))
plot_tree(dt, feature_names=fnames, class_names=["-", "+"], filled=True, rounded=True, fontsize=8, impurity=False, proportion=False)
savefig("13_decision_tree")
imp_dt = pd.Series(dt.feature_importances_, index=fnames).sort_values(ascending=False)
imp_dt = imp_dt[imp_dt > 0]
save_table("dt_importance", imp_dt.round(3).rename("Importance").reset_index().rename(columns={"index": "Encoded feature"}))

# 5.2 rule list on all data
rl = SequentialCoveringRules().fit(make_pre("mean_mode", rules=True).fit_transform(X), y)
rule_tbl = pd.DataFrame([[a, b, c, d, f"{100 * e:.1f}%"] for a, b, c, d, e in rl.describe()],
                        columns=["Rule", "IF (conditions)", "Positives covered", "Negatives covered", "Precision"])
rule_tbl["THEN"] = "Class +"
save_table("rule_list", rule_tbl)
NUM["n_rules"] = len(rl.rules_)
NUM["rule_cov_pos"] = int(sum(r[1] for r in rl.rules_))

# 5.3 out-of-fold predictions for audit (decision tree and others)
cv_a = StratifiedKFold(10, shuffle=True, random_state=SEED)
models_oof = {"Decision tree": build_model("DT", "mean_mode", "onehot", "zscore"),
              "Rule-based": build_model("RULES", "mean_mode", "native", "none"),
              "KNN": build_model("KNN", "mean_mode", "onehot", "zscore")}
OOF = {k: cross_val_predict(m, X, y, cv=cv_a, method="predict_proba")[:, 1] for k, m in models_oof.items()}
np.save(f"{TAB}/oof_dt_proba.npy", OOF["Decision tree"])

plt.figure(figsize=(4.6, 3.8))
for k, pr in OOF.items():
    fpr, tpr, _ = roc_curve(y, pr)
    plt.plot(fpr, tpr, label=f"{k} (AUC {roc_auc_score(y, pr):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.legend(loc="lower right", fontsize=8); plt.title("ROC curves (out-of-fold predictions)")
savefig("14_roc")

rows = []
for k, pr in OOF.items():
    tn, fp, fn, tp = confusion_matrix(y, pr >= .5).ravel()
    rows.append([k, tn, fp, fn, tp])
save_table("confusion_matrices", pd.DataFrame(rows, columns=["Classifier", "TN", "FP", "FN", "TP"]))

# threshold trade-off for the decision tree
rows = []
p = OOF["Decision tree"]
for th in (0.2, 0.35, 0.5, 0.65, 0.8):
    pr = p >= th
    tn, fp, fn, tp = confusion_matrix(y, pr).ravel()
    rows.append([th, 100 * (tp + tn) / len(y), 100 * tp / (tp + fn), 100 * fp / (fp + tn), 100 * fp / max(fp + tp, 1)])
th_df = pd.DataFrame(rows, columns=["Threshold for class +", "Accuracy %", "True positive rate %", "False positive rate %", "Share of predicted + that are wrong %"])
th_df.iloc[:, 1:] = th_df.iloc[:, 1:].round(1)
save_table("threshold_tradeoff", th_df)

# audit by anonymous attribute A1 (meaning not documented)
rows = []
pr = (OOF["Decision tree"] >= .5).astype(int)
for grp in ("a", "b", "missing"):
    mask = (X["A1"].fillna("missing") == grp).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y[mask], pr[mask], labels=[0, 1]).ravel()
    rows.append([f"A1 = {grp}", int(mask.sum()), 100 * y[mask].mean(), 100 * pr[mask].mean(),
                 100 * tp / max(tp + fn, 1), 100 * fp / max(fp + tn, 1), 100 * (tp + tn) / mask.sum()])
save_table("audit_A1", pd.DataFrame(rows, columns=["Group", "n", "Actual + %", "Predicted + %", "TPR %", "FPR %", "Accuracy %"]).round(1))

# misclassification for rows with / without missing values
miss_mask = X.isna().any(axis=1).to_numpy()
rows = []
for lab, mk in (("Complete rows", ~miss_mask), ("Rows with missing values", miss_mask)):
    rows.append([lab, int(mk.sum()), 100 * (pr[mk] == y[mk]).mean()])
save_table("audit_missing", pd.DataFrame(rows, columns=["Group", "n", "Accuracy %"]).round(1))
print("Part 3 done")
# %% [markdown]
# # Part 4: Frequent itemsets and association rules (Module 3b)
# Apriori and FP-Growth are implemented from scratch (mlxtend is not required) so that every
# algorithmic step can be explained.

# %% [markdown]
# ## 1. Discretisation and transaction database

# %%
def discretize(s, name):
    """Equal-frequency 3 bins; zero-inflated attributes get a separate 'zero' bin."""
    out = pd.Series(np.nan, index=s.index, dtype=object)
    if (s == 0).mean() > 0.25:
        cut = s[s > 0].median()
        out[s == 0] = f"{name}=zero"
        out[(s > 0) & (s <= cut)] = f"{name}=low"
        out[s > cut] = f"{name}=high"
        desc = f"zero: 0;  low: 0 < x <= {cut:g};  high: x > {cut:g}"
    else:
        q1, q2 = s.quantile([1 / 3, 2 / 3])
        out[s <= q1] = f"{name}=low"
        out[(s > q1) & (s <= q2)] = f"{name}=mid"
        out[s > q2] = f"{name}=high"
        desc = f"low: x <= {q1:g};  mid: {q1:g} < x <= {q2:g};  high: x > {q2:g}"
    out[s.isna()] = np.nan
    return out, desc

DISC = pd.DataFrame(index=X.index)
cut_rows = []
for c in NUM_COLS:
    DISC[c], d = discretize(X[c], c)
    cut_rows.append([c, d])
save_table("discretization_cuts", pd.DataFrame(cut_rows, columns=["Attribute", "Bins (equal-frequency / zero-inflated)"]))

ITEM_CATS = ["A1", "A4", "A9", "A12", "A13"]   # A5 duplicates A4, A10 duplicates (A11 > 0); A6, A7 have too many sparse values
trans = []
for i in range(len(X)):
    items = [DISC.loc[i, c] for c in NUM_COLS if pd.notna(DISC.loc[i, c])]
    items += [f"{c}={X.loc[i, c]}" for c in ITEM_CATS if pd.notna(X.loc[i, c])]
    items.append("Class=+" if y[i] == 1 else "Class=-")
    trans.append(frozenset(items))
N_TRANS = len(trans)
NUM["n_items"] = len({i for t in trans for i in t})
NUM["avg_items_per_trans"] = float(np.mean([len(t) for t in trans]))


# %% [markdown]
# ## 2. Apriori (level-wise candidate generation, prune by the Apriori property)
# Support counting uses transaction-id bit sets (one Python integer per item): the support of a
# candidate is the popcount of the AND of the bit sets of its two parents.

# %%
def popcount(x):
    return x.bit_count() if hasattr(x, "bit_count") else bin(x).count("1")


def build_tidsets(trans):
    tid = defaultdict(int)
    for k, t in enumerate(trans):
        for i in t:
            tid[i] |= 1 << k
    return tid


def apriori(trans, min_sup):
    n = len(trans)
    min_count = math.ceil(min_sup * n)
    tid = build_tidsets(trans)
    L = {(i,): m for i, m in tid.items() if popcount(m) >= min_count}
    freq = {frozenset(k): popcount(m) for k, m in L.items()}
    n_candidates = len(tid)
    levels = [(1, len(tid), len(L))]
    k = 2
    while L:
        keys = sorted(L)
        keyset = set(keys)
        newL, ncand = {}, 0
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if keys[a][:-1] != keys[b][:-1]:
                    break                                   # join step: same (k-2)-prefix
                cand = keys[a] + (keys[b][-1],)
                if all(cand[:j] + cand[j + 1:] in keyset for j in range(len(cand) - 2)):   # prune step
                    ncand += 1
                    m = L[keys[a]] & L[keys[b]]
                    if popcount(m) >= min_count:
                        newL[cand] = m
        for cand, m in newL.items():
            freq[frozenset(cand)] = popcount(m)
        levels.append((k, ncand, len(newL)))
        n_candidates += ncand
        L = newL
        k += 1
    return freq, n_candidates, levels


# %% [markdown]
# ## 3. FP-Growth (FP-tree construction and recursive mining of conditional pattern bases)

# %%
class FPNode:
    __slots__ = ("item", "count", "parent", "children")
    def __init__(self, item, parent):
        self.item, self.count, self.parent, self.children = item, 0, parent, {}


def build_fptree(weighted, min_count):
    cnt = Counter()
    for items, w in weighted:
        for i in items:
            cnt[i] += w
    freq = {i: c for i, c in cnt.items() if c >= min_count}
    order = {i: r for r, (i, _) in enumerate(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))}
    root, header = FPNode(None, None), defaultdict(list)
    for items, w in weighted:
        node = root
        for i in sorted((i for i in items if i in freq), key=order.get):
            ch = node.children.get(i)
            if ch is None:
                ch = FPNode(i, node)
                node.children[i] = ch
                header[i].append(ch)
            ch.count += w
            node = ch
    return header, freq


def fp_growth(trans, min_sup):
    n = len(trans)
    min_count = math.ceil(min_sup * n)
    header, freq0 = build_fptree([(t, 1) for t in trans], min_count)
    result = {}

    def mine(header, freq, suffix):
        for item in sorted(freq, key=lambda i: (freq[i], i)):          # least frequent first
            new = suffix | {item}
            result[frozenset(new)] = sum(nd.count for nd in header[item])
            cpb = []                                                    # conditional pattern base
            for nd in header[item]:
                path, p = [], nd.parent
                while p.item is not None:
                    path.append(p.item); p = p.parent
                if path:
                    cpb.append((path, nd.count))
            if cpb:
                h2, f2 = build_fptree(cpb, min_count)
                if f2:
                    mine(h2, f2, new)
    mine(header, freq0, frozenset())
    return result


# %% [markdown]
# ## 4. Run both algorithms at several minimum supports and verify they agree

# %%
def best_of(fn, reps=3):
    times, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t0)
    return out, min(times)

rows = []
RESULTS_FREQ = {}
for ms_ in (0.40, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05):
    (fa, ncand, levels), ta = best_of(lambda: apriori(trans, ms_))
    ff, tf = best_of(lambda: fp_growth(trans, ms_))
    RESULTS_FREQ[ms_] = fa
    rows.append([ms_, len(fa), max(len(s_) for s_ in fa), ncand, round(ta, 3), round(tf, 3), "yes" if fa == ff else "NO"])
TIMING = pd.DataFrame(rows, columns=[
    "Min. support", "Frequent itemsets", "Longest itemset", "Apriori candidates tested",
    "Apriori time (s)", "FP-Growth time (s)", "Identical results"])
save_table("apriori_vs_fpgrowth", TIMING)
NUM["assoc_identical"] = bool(all(r[-1] == "yes" for r in rows))

# per-level counts for the main support threshold
MAIN_SUP = 0.15
fa, ncand, levels = apriori(trans, MAIN_SUP)
save_table("apriori_levels", pd.DataFrame(levels, columns=["Level k", "Candidates tested", "Frequent k-itemsets"]))
NUM["main_sup"], NUM["main_itemsets"] = MAIN_SUP, len(fa)


# %% [markdown]
# ## 5. Association rules (single-item consequent) and interestingness measures

# %%
def gen_rules(freq, n, min_conf):
    rules = []
    for itemset, cnt in freq.items():
        if len(itemset) < 2:
            continue
        for cons in itemset:
            ante = itemset - {cons}
            ca, cc = freq[ante], freq[frozenset([cons])]
            conf = cnt / ca
            if conf >= min_conf:
                sup = cnt / n
                lift = conf / (cc / n)
                lev = sup - (ca / n) * (cc / n)
                conv = (1 - cc / n) / (1 - conf) if conf < 1 else np.inf
                rules.append((tuple(sorted(ante)), cons, sup, conf, lift, lev, conv, cnt))
    return pd.DataFrame(rules, columns=["antecedent", "consequent", "support", "confidence", "lift", "leverage", "conviction", "count"])

def non_redundant(df):
    """Drop a rule if a simpler rule (proper subset of its antecedent) with the same consequent is at least as confident."""
    groups = {c: [(set(a), cf) for a, cf in zip(g.antecedent, g.confidence)] for c, g in df.groupby("consequent")}
    keep = []
    for r in df.itertuples():
        A = set(r.antecedent)
        keep.append(not any((B < A) and (cf >= r.confidence - 1e-9) for B, cf in groups[r.consequent]))
    return df[np.array(keep)]

R_all = gen_rules(fa, N_TRANS, 0.80)
R = non_redundant(R_all)
NUM["n_rules_all_conf80"] = int(len(R_all))
NUM["n_rules_conf80"] = int(len(R))
NUM["n_rules_lift"] = int((R["lift"] > 1.2).sum())

def rule_text(r):
    return " AND ".join(r.antecedent) + "  →  " + r.consequent

def top_rules(df, cons, k=8, min_lift=1.0):
    d = df[(df.consequent == cons) & (df.lift >= min_lift)].copy()
    d = d.sort_values(["lift", "confidence", "support"], ascending=False).head(k)
    out = pd.DataFrame({"Rule": [rule_text(r) for r in d.itertuples()],
                        "Support": d.support.round(3), "Confidence": d.confidence.round(3),
                        "Lift": d.lift.round(2), "Leverage": d.leverage.round(3)})
    return out

save_table("rules_class_pos", top_rules(R, "Class=+"))
save_table("rules_class_neg", top_rules(R, "Class=-"))
gen = R[~R.consequent.str.startswith("Class") & ~R.antecedent.map(lambda a: any(i.startswith("Class") for i in a))]
g = gen.sort_values(["lift", "confidence"], ascending=False).head(8)
save_table("rules_general", pd.DataFrame({"Rule": [rule_text(r) for r in g.itertuples()],
                                          "Support": g.support.round(3), "Confidence": g.confidence.round(3),
                                          "Lift": g.lift.round(2), "Leverage": g.leverage.round(3)}))
R_all.sort_values("lift", ascending=False).head(500).to_csv(f"{TAB}/association_rules_top500.csv", index=False)

# effect of thresholds on the number of class rules
rows = []
for ms_ in (0.30, 0.20, 0.15, 0.10):
    Ra = gen_rules(RESULTS_FREQ[ms_], N_TRANS, 0.8)
    Rm = non_redundant(Ra)
    rows.append([ms_, len(RESULTS_FREQ[ms_]), len(Ra), len(Rm), int((Rm.consequent == "Class=+").sum()), int((Rm.consequent == "Class=-").sum())])
save_table("rules_threshold_effect", pd.DataFrame(rows, columns=["Min. support", "Frequent itemsets", "Rules (conf >= 0.8)", "Non-redundant rules", "Class=+ rules", "Class=- rules"]))

plt.figure(figsize=(5, 3.4))
Rs = R.sample(min(len(R), 3000), random_state=SEED)
sc_ = plt.scatter(Rs.support, Rs.confidence, c=Rs.lift, cmap="viridis", s=8, alpha=.7)
plt.colorbar(sc_, label="Lift"); plt.xlabel("Support"); plt.ylabel("Confidence"); plt.title("Association rules (support, confidence, lift)")
savefig("15_rules_scatter")

plt.figure(figsize=(5, 3.2))
plt.plot(TIMING["Min. support"], TIMING["Apriori time (s)"].clip(lower=1e-3), "o-", label="Apriori")
plt.plot(TIMING["Min. support"], TIMING["FP-Growth time (s)"].clip(lower=1e-3), "s-", label="FP-Growth")
plt.gca().invert_xaxis(); plt.yscale("log"); plt.xlabel("Minimum support"); plt.ylabel("Run time (s, log scale)")
plt.legend(); plt.title("Apriori versus FP-Growth run time")
savefig("16_apriori_fpgrowth_time")
print("Part 4 done")
# %% [markdown]
# # Part 5: Clustering (Module 4)
# K-Means, hierarchical clustering and DBSCAN on three representations of the mixed data:
#   R1 = numeric attributes only (Z-score);  R2 = numeric (Z-score) + one-hot nominal;
#   R3 = Gower dissimilarity matrix (native mixed-type distance, missing values skipped).

# %%
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, normalized_mutual_info_score)
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors

Zc = make_pre("mean_mode", "onehot", "zscore").fit_transform(X)      # R2
Zn = Zc[:, :len(NUM_COLS)]                                            # R1 (numeric block comes first)
NUM["n_cluster_dims"] = int(Zc.shape[1])
Gd = G.copy(); np.fill_diagonal(Gd, 0)


def purity(labels):
    ok = labels >= 0
    if ok.sum() == 0:
        return np.nan
    tot = 0
    for c in np.unique(labels[ok]):
        tot += np.bincount(y[labels == c]).max()
    return tot / ok.sum()


def internal_external(space, labels, metric="euclidean"):
    ok = labels >= 0
    n_cl = len(np.unique(labels[ok]))
    out = {"clusters": n_cl, "noise": int((~ok).sum())}
    if n_cl >= 2 and ok.sum() > n_cl:
        if metric == "precomputed":
            out["sil"] = silhouette_score(space[np.ix_(ok, ok)], labels[ok], metric="precomputed")
            out["db"], out["ch"] = np.nan, np.nan
        else:
            out["sil"] = silhouette_score(space[ok], labels[ok])
            out["db"] = davies_bouldin_score(space[ok], labels[ok])
            out["ch"] = calinski_harabasz_score(space[ok], labels[ok])
    else:
        out["sil"] = out["db"] = out["ch"] = np.nan
    out["ari"] = adjusted_rand_score(y, labels)
    out["nmi"] = normalized_mutual_info_score(y, labels)
    out["purity"] = purity(labels)
    return out


# %% [markdown]
# ## 1. K-Means (k = 2 ... 8) on R1 and R2

# %%
rows, SSE = [], {}
KM_LABELS = {}
for rep, space in (("R1 numeric Z-score", Zn), ("R2 one-hot + Z-score", Zc)):
    SSE[rep] = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(space)
        KM_LABELS[(rep, k)] = km.labels_
        m = internal_external(space, km.labels_)
        SSE[rep].append(km.inertia_)
        rows.append([rep, k, km.inertia_, m["sil"], m["db"], m["ch"], m["ari"], m["nmi"], m["purity"]])
km_df = pd.DataFrame(rows, columns=["Representation", "k", "SSE", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz",
                                    "ARI vs class", "NMI vs class", "Purity"])
save_table("kmeans_results", km_df.round(3))
best_row = km_df[km_df.Representation == "R2 one-hot + Z-score"].sort_values("Silhouette", ascending=False).iloc[0]
K_BEST = int(best_row.k)
NUM["km_best_k"], NUM["km_best_sil"] = K_BEST, float(best_row.Silhouette)
NUM["km_k2_sil"] = float(km_df[(km_df.Representation == "R2 one-hot + Z-score") & (km_df.k == 2)].Silhouette.iloc[0])
NUM["km_k2_ari"] = float(km_df[(km_df.Representation == "R2 one-hot + Z-score") & (km_df.k == 2)]["ARI vs class"].iloc[0])
NUM["km_k2_purity"] = float(km_df[(km_df.Representation == "R2 one-hot + Z-score") & (km_df.k == 2)].Purity.iloc[0])

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for rep in SSE:
    axes[0].plot(range(2, 9), SSE[rep], "o-", ms=3, label=rep)
    axes[1].plot(range(2, 9), km_df[km_df.Representation == rep].Silhouette, "o-", ms=3, label=rep)
axes[0].set_title("Elbow curve (SSE)"); axes[0].set_xlabel("k"); axes[0].set_yscale("log")
axes[1].set_title("Silhouette coefficient"); axes[1].set_xlabel("k"); axes[1].legend(fontsize=7)
plt.tight_layout(); savefig("17_kmeans_k_selection")

# %% [markdown]
# ## 2. Hierarchical clustering

# %%
rows = []
HC_LABELS = {}
for name, link_, space, metric in (("Ward on R2", "ward", Zc, "euclidean"),
                                   ("Average linkage on Gower", "average", Gd, "precomputed"),
                                   ("Complete linkage on Gower", "complete", Gd, "precomputed")):
    for k in range(2, 7):
        if metric == "precomputed":
            lab = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=link_).fit_predict(space)
        else:
            lab = AgglomerativeClustering(n_clusters=k, linkage=link_).fit_predict(space)
        HC_LABELS[(name, k)] = lab
        m = internal_external(space, lab, metric)
        rows.append([name, k, m["sil"], m["ari"], m["nmi"], m["purity"], int(np.bincount(lab).min())])
hc_df = pd.DataFrame(rows, columns=["Method", "k", "Silhouette", "ARI vs class", "NMI vs class", "Purity", "Smallest cluster"])
save_table("hierarchical_results", hc_df.round(3))

# cophenetic correlation: how faithfully each dendrogram preserves the original distances
D_euc = pairwise_distances(Zc)
coph = []
for name, link_, dmat in (("Ward on R2 (Euclidean)", "ward", D_euc), ("Average on Gower", "average", Gd),
                          ("Complete on Gower", "complete", Gd), ("Single on Gower", "single", Gd)):
    Lk = linkage(squareform(dmat, checks=False), method=link_)
    coph.append([name, cophenet(Lk, squareform(dmat, checks=False))[0]])
save_table("cophenetic", pd.DataFrame(coph, columns=["Linkage", "Cophenetic correlation"]).round(3))

Lw = linkage(Zc, method="ward")
plt.figure(figsize=(7, 3.4))
dendrogram(Lw, truncate_mode="lastp", p=24, no_labels=True, color_threshold=0.7 * Lw[-1, 2])
plt.title("Ward dendrogram on R2 (truncated to 24 leaves)"); plt.ylabel("Merge distance")
savefig("18_dendrogram")

# %% [markdown]
# ## 3. DBSCAN

# %%
kd_c = np.sort(NearestNeighbors(n_neighbors=6).fit(Zc).kneighbors(Zc)[0][:, -1])
kd_g = np.sort(np.sort(Gd, axis=1)[:, 6])
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].plot(kd_c); axes[0].set_title("5-NN distance, R2 (Euclidean)"); axes[0].set_ylabel("Distance to 5th neighbour")
axes[1].plot(kd_g); axes[1].set_title("5-NN distance, R3 (Gower)")
for a in axes: a.set_xlabel("Points sorted by distance")
plt.tight_layout(); savefig("19_kdistance")

rows = []
DB_LABELS = {}
for rep, space, metric, epss in (("R2 one-hot + Z-score", Zc, "euclidean", np.round(np.quantile(kd_c, [.5, .7, .8, .9, .95]), 2)),
                                 ("R3 Gower", Gd, "precomputed", [0.10, 0.15, 0.20, 0.25, 0.30])):
    for eps in epss:
        for mp_ in (4, 6, 10):
            lab = DBSCAN(eps=float(eps), min_samples=mp_, metric=metric).fit_predict(space)
            DB_LABELS[(rep, float(eps), mp_)] = lab
            m = internal_external(space, lab, metric)
            rows.append([rep, float(eps), mp_, m["clusters"], m["noise"], m["sil"], m["ari"], m["purity"]])
db_df = pd.DataFrame(rows, columns=["Representation", "eps", "MinPts", "Clusters", "Noise points", "Silhouette (non-noise)", "ARI vs class", "Purity (non-noise)"])
save_table("dbscan_grid", db_df.round(3))
cand = db_df[(db_df.Clusters >= 2) & (db_df["Noise points"] <= 0.30 * len(X))]
if len(cand):
    best_db = cand.sort_values("Silhouette (non-noise)", ascending=False).iloc[0]
else:
    best_db = db_df[db_df.Clusters >= 1].sort_values("Noise points").iloc[0]
DB_BEST_KEY = (best_db.Representation, float(best_db.eps), int(best_db.MinPts))
DB_BEST = DB_LABELS[DB_BEST_KEY]
NUM["db_best"] = {"rep": best_db.Representation, "eps": float(best_db.eps), "minpts": int(best_db.MinPts),
                  "clusters": int(best_db.Clusters), "noise": int(best_db["Noise points"]),
                  "sil": None if pd.isna(best_db["Silhouette (non-noise)"]) else float(best_db["Silhouette (non-noise)"]),
                  "ari": float(best_db["ARI vs class"])}
NUM["db_configs_with_ge2_clusters"] = int((db_df.Clusters >= 2).sum()); NUM["db_configs_total"] = int(len(db_df))

# %% [markdown]
# ## 4. Visual comparison and cluster profiles

# %%
P2 = PCA(n_components=2, random_state=SEED).fit_transform(Zc)
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].scatter(P2[:, 0], P2[:, 1], c=np.where(y == 1, "#55A868", "#C44E52"), s=7, alpha=.65); axes[0].set_title("True class (+ green, - red)")
axes[1].scatter(P2[:, 0], P2[:, 1], c=KM_LABELS[("R2 one-hot + Z-score", K_BEST)], cmap="tab10", s=7, alpha=.7); axes[1].set_title(f"K-Means, k = {K_BEST}")
lab = DB_BEST
axes[2].scatter(P2[lab < 0, 0], P2[lab < 0, 1], c="lightgrey", s=7)
axes[2].scatter(P2[lab >= 0, 0], P2[lab >= 0, 1], c=lab[lab >= 0], cmap="tab10", s=7, alpha=.8); axes[2].set_title("DBSCAN (noise in grey)")
for a in axes: a.set_xlabel("PC1"); a.set_ylabel("PC2")
plt.tight_layout(); savefig("20_cluster_pca_views")

def profile(labels, name):
    d = Xm.copy(); d["cluster"] = labels; d["pos"] = y
    g = d.groupby("cluster")
    out = pd.DataFrame({"Size": g.size(), "Class + %": 100 * g.pos.mean(),
                        "A9=t %": 100 * g.apply(lambda s: (s.A9 == "t").mean()),
                        "A10=t %": 100 * g.apply(lambda s: (s.A10 == "t").mean()),
                        "Mean A2": g.A2.mean(), "Mean A3": g.A3.mean(), "Mean A8": g.A8.mean(),
                        "Mean A11": g.A11.mean(), "Median A15": g.A15.median()})
    return out.round(1).reset_index().rename(columns={"cluster": name})

save_table("profile_kmeans_k2", profile(KM_LABELS[("R2 one-hot + Z-score", 2)], "K-Means cluster"))
save_table("profile_kmeans_best", profile(KM_LABELS[("R2 one-hot + Z-score", K_BEST)], "K-Means cluster"))
lab_db = DB_BEST.copy()
save_table("profile_dbscan", profile(lab_db, "DBSCAN cluster (-1 = noise)"))
print("Part 5 done")
# %% [markdown]
# # Part 6: Data warehouse and OLAP (Module 2)
# A fact-constellation (galaxy) schema in SQLite with three fact tables:
#   fact_application        - one row per credit application (application characteristics and outcome)
#   fact_experiment_run     - one row per classification experiment (model x preprocessing)
#   fact_feature_importance - one row per (feature, selection method)
# Dimensions are shared where possible; dim_feature_group links features to the documented attribute groups.

# %%
DB_PATH = f"{WH}/credit_dw.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

DDL = """
CREATE TABLE dim_outcome (outcome_key INTEGER PRIMARY KEY, outcome_label TEXT NOT NULL);
CREATE TABLE dim_quality (quality_key INTEGER PRIMARY KEY, quality_label TEXT NOT NULL, min_missing INTEGER);
CREATE TABLE dim_profile (profile_key INTEGER PRIMARY KEY, a1 TEXT, a4 TEXT, a5 TEXT, a9 TEXT, a10 TEXT, a12 TEXT, a13 TEXT);
CREATE TABLE dim_a6 (a6_key INTEGER PRIMARY KEY, a6_value TEXT NOT NULL);
CREATE TABLE dim_a7 (a7_key INTEGER PRIMARY KEY, a7_value TEXT NOT NULL);
CREATE TABLE dim_band (band_key INTEGER PRIMARY KEY, band_label TEXT NOT NULL);
CREATE TABLE dim_cluster (cluster_key INTEGER PRIMARY KEY, algorithm TEXT NOT NULL, cluster_label TEXT NOT NULL);
CREATE TABLE dim_feature_group (group_key INTEGER PRIMARY KEY, group_name TEXT NOT NULL, description TEXT);
CREATE TABLE dim_feature (feature_key INTEGER PRIMARY KEY, feature_name TEXT NOT NULL, group_key INTEGER NOT NULL,
                          n_distinct INTEGER, n_missing INTEGER,
                          FOREIGN KEY (group_key) REFERENCES dim_feature_group(group_key));
CREATE TABLE dim_selection_method (method_key INTEGER PRIMARY KEY, method_name TEXT NOT NULL);
CREATE TABLE dim_model (model_key INTEGER PRIMARY KEY, model_name TEXT NOT NULL, model_family TEXT);
CREATE TABLE dim_preprocessing (prep_key INTEGER PRIMARY KEY, missing_strategy TEXT, encoding TEXT, scaling TEXT,
                                reduction TEXT, k_features INTEGER);

CREATE TABLE fact_application (
    app_id INTEGER PRIMARY KEY,
    outcome_key INTEGER NOT NULL, profile_key INTEGER NOT NULL, a6_key INTEGER NOT NULL, a7_key INTEGER NOT NULL,
    quality_key INTEGER NOT NULL, cluster_key INTEGER NOT NULL,
    a2_band_key INTEGER, a3_band_key INTEGER, a8_band_key INTEGER, a11_band_key INTEGER, a14_band_key INTEGER, a15_band_key INTEGER,
    a2 REAL, a3 REAL, a8 REAL, a11 REAL, a14 REAL, a15 REAL,
    missing_count INTEGER, is_positive INTEGER, cv_pred_prob REAL, cv_pred_class INTEGER, is_correct INTEGER,
    FOREIGN KEY (outcome_key) REFERENCES dim_outcome(outcome_key),
    FOREIGN KEY (profile_key) REFERENCES dim_profile(profile_key),
    FOREIGN KEY (a6_key) REFERENCES dim_a6(a6_key),
    FOREIGN KEY (a7_key) REFERENCES dim_a7(a7_key),
    FOREIGN KEY (quality_key) REFERENCES dim_quality(quality_key),
    FOREIGN KEY (cluster_key) REFERENCES dim_cluster(cluster_key),
    FOREIGN KEY (a2_band_key) REFERENCES dim_band(band_key), FOREIGN KEY (a3_band_key) REFERENCES dim_band(band_key),
    FOREIGN KEY (a8_band_key) REFERENCES dim_band(band_key), FOREIGN KEY (a11_band_key) REFERENCES dim_band(band_key),
    FOREIGN KEY (a14_band_key) REFERENCES dim_band(band_key), FOREIGN KEY (a15_band_key) REFERENCES dim_band(band_key));

CREATE TABLE fact_experiment_run (
    run_key INTEGER PRIMARY KEY, model_key INTEGER NOT NULL, prep_key INTEGER NOT NULL, n_rows INTEGER,
    acc_mean REAL, acc_std REAL, prec_mean REAL, rec_mean REAL, f1_mean REAL, auc_mean REAL, time_s REAL,
    FOREIGN KEY (model_key) REFERENCES dim_model(model_key), FOREIGN KEY (prep_key) REFERENCES dim_preprocessing(prep_key));

CREATE TABLE fact_feature_importance (
    feature_key INTEGER NOT NULL, method_key INTEGER NOT NULL, score REAL, rank_in_method INTEGER,
    PRIMARY KEY (feature_key, method_key),
    FOREIGN KEY (feature_key) REFERENCES dim_feature(feature_key),
    FOREIGN KEY (method_key) REFERENCES dim_selection_method(method_key));
"""
open(f"{WH}/schema.sql", "w").write(DDL)
con = sqlite3.connect(DB_PATH)
con.executescript(DDL)

# %% [markdown]
# ## 1. Extract-transform-load (ETL)

# %%
def load(table, df):
    df.to_sql(table, con, if_exists="append", index=False)

load("dim_outcome", pd.DataFrame({"outcome_key": [1, 2], "outcome_label": ["+", "-"]}))
load("dim_quality", pd.DataFrame({"quality_key": [1, 2, 3], "quality_label": ["Complete", "One missing value", "Two or more missing"],
                                  "min_missing": [0, 1, 2]}))
band_labels = ["Zero", "Low", "Mid", "High", "Missing"]
load("dim_band", pd.DataFrame({"band_key": range(1, 6), "band_label": band_labels}))
for tbl, col in (("dim_a6", "a6"), ("dim_a7", "a7")):
    vals = sorted(X[col.upper()].dropna().unique()) + ["Missing"]
    load(tbl, pd.DataFrame({f"{col}_key": range(1, len(vals) + 1), f"{col}_value": vals}))
a6_map = {v: i + 1 for i, v in enumerate(sorted(X["A6"].dropna().unique()) + ["Missing"])}
a7_map = {v: i + 1 for i, v in enumerate(sorted(X["A7"].dropna().unique()) + ["Missing"])}

prof_cols = ["A1", "A4", "A5", "A9", "A10", "A12", "A13"]
prof = X[prof_cols].fillna("Missing")
combos = prof.drop_duplicates().reset_index(drop=True)
combos["profile_key"] = combos.index + 1
load("dim_profile", combos.rename(columns=str.lower)[["profile_key", "a1", "a4", "a5", "a9", "a10", "a12", "a13"]])
profile_key = prof.merge(combos, on=prof_cols, how="left")["profile_key"].to_numpy()

km_lab = KM_LABELS[("R2 one-hot + Z-score", K_BEST)]
load("dim_cluster", pd.DataFrame({"cluster_key": range(1, K_BEST + 1), "algorithm": f"K-Means (k={K_BEST}, one-hot + Z-score)",
                                  "cluster_label": [f"C{i}" for i in range(K_BEST)]}))

band_key_cols = {}
for c in NUM_COLS:
    lab = DISC[c].fillna("Missing").map(lambda s: s.split("=")[1] if "=" in s else s).str.capitalize()
    band_key_cols[c] = lab.map({b: i + 1 for i, b in enumerate(band_labels)}).to_numpy()

miss_cnt = X.isna().sum(axis=1).to_numpy()
quality_key = np.where(miss_cnt == 0, 1, np.where(miss_cnt == 1, 2, 3))
p_dt = np.load(f"{TAB}/oof_dt_proba.npy")
fa_df = pd.DataFrame({
    "app_id": np.arange(1, len(X) + 1),
    "outcome_key": np.where(y == 1, 1, 2), "profile_key": profile_key,
    "a6_key": X["A6"].fillna("Missing").map(a6_map).to_numpy(), "a7_key": X["A7"].fillna("Missing").map(a7_map).to_numpy(),
    "quality_key": quality_key, "cluster_key": km_lab + 1,
    **{f"{c.lower()}_band_key": band_key_cols[c] for c in ["A2", "A3", "A8", "A11", "A14", "A15"]},
    **{c.lower(): X[c].to_numpy() for c in ["A2", "A3", "A8", "A11", "A14", "A15"]},
    "missing_count": miss_cnt, "is_positive": y, "cv_pred_prob": p_dt.round(4),
    "cv_pred_class": (p_dt >= .5).astype(int), "is_correct": ((p_dt >= .5).astype(int) == y).astype(int)})
load("fact_application", fa_df)

# feature dimensions
grp = {"Continuous": ["A2", "A3", "A8", "A11", "A14", "A15"],
       "Nominal, small number of values": ["A1", "A4", "A5", "A9", "A10", "A12", "A13"],
       "Nominal, larger number of values": ["A6", "A7"]}
grp_desc = {"Continuous": "Documented as continuous (A11, A14, A15 are integer valued in the file)",
            "Nominal, small number of values": "2 to 4 values each (A1, A9, A10, A12 are binary)",
            "Nominal, larger number of values": "9 and 14 values"}
load("dim_feature_group", pd.DataFrame({"group_key": [1, 2, 3], "group_name": list(grp), "description": list(grp_desc.values())}))
gkey = {a: i + 1 for i, (g, lst) in enumerate(grp.items()) for a in lst}
feat_rows = [[i + 1, c, gkey[c], int(X[c].nunique()), int(X[c].isna().sum())] for i, c in enumerate(X.columns)]
load("dim_feature", pd.DataFrame(feat_rows, columns=["feature_key", "feature_name", "group_key", "n_distinct", "n_missing"]))
fkey = {c: i + 1 for i, c in enumerate(X.columns)}
meths = ["Mutual information", "-log10 p-value", "Random-forest importance"]
load("dim_selection_method", pd.DataFrame({"method_key": [1, 2, 3], "method_name": meths}))
sel_raw = sel[meths]
rows = []
for mi_, m in enumerate(meths, 1):
    rk = sel_raw[m].rank(ascending=False, method="min").astype(int)
    for a in X.columns:
        rows.append([fkey[a], mi_, float(sel_raw.loc[a, m]), int(rk[a])])
load("fact_feature_importance", pd.DataFrame(rows, columns=["feature_key", "method_key", "score", "rank_in_method"]))

# experiments
mname = {"DT": ("Decision tree (depth 4)", "Tree"), "RULES": ("Rule-based (sequential covering)", "Rules"), "KNN": ("KNN (k=7)", "Instance-based")}
load("dim_model", pd.DataFrame({"model_key": [1, 2, 3], "model_name": [mname[k][0] for k in ("DT", "RULES", "KNN")],
                                "model_family": [mname[k][1] for k in ("DT", "RULES", "KNN")]}))
mkey = {"DT": 1, "RULES": 2, "KNN": 3}
preps = exp_df[["missing", "encoding", "scaling", "reduction", "k_feat"]].drop_duplicates().reset_index(drop=True)
preps["prep_key"] = preps.index + 1
load("dim_preprocessing", preps.rename(columns={"missing": "missing_strategy", "k_feat": "k_features"})
     [["prep_key", "missing_strategy", "encoding", "scaling", "reduction", "k_features"]])
ex = exp_df.merge(preps, on=["missing", "encoding", "scaling", "reduction", "k_feat"])
ex["run_key"] = np.arange(1, len(ex) + 1)
ex["model_key"] = ex.model.map(mkey)
load("fact_experiment_run", ex.rename(columns={"acc_mean": "acc_mean"})[["run_key", "model_key", "prep_key", "n_rows", "acc_mean", "acc_std",
                                                                       "prec_mean", "rec_mean", "f1_mean", "auc_mean", "time_s"]])
con.commit()
cnt = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
       ["fact_application", "fact_experiment_run", "fact_feature_importance", "dim_profile", "dim_feature", "dim_preprocessing"]}
NUM["wh_counts"] = cnt
save_table("warehouse_row_counts", pd.DataFrame([[k, v] for k, v in cnt.items()], columns=["Table", "Rows"]))

# %% [markdown]
# ## 2. OLAP operations expressed in SQL
# SQLite has no ROLLUP / CUBE, so roll-up sub-totals are produced with UNION ALL.

# %%
APP_JOIN = """FROM fact_application f
JOIN dim_profile p ON f.profile_key = p.profile_key
JOIN dim_outcome o ON f.outcome_key = o.outcome_key"""
QUERIES = [
 ("Q1", "Roll-up", "What is the share of positive-class applications for each value of A9, and overall?",
  f"""SELECT p.a9 AS a9, COUNT(*) AS applications, SUM(f.is_positive) AS positives,
       ROUND(100.0 * SUM(f.is_positive) / COUNT(*), 1) AS positive_pct
{APP_JOIN}
GROUP BY p.a9
UNION ALL
SELECT 'ALL', COUNT(*), SUM(is_positive), ROUND(100.0 * SUM(is_positive) / COUNT(*), 1) FROM fact_application
ORDER BY 1;"""),
 ("Q2", "Drill-down", "Drilling down from A9 to the combination A9 x A10 x A13, where do positives concentrate?",
  f"""SELECT p.a9, p.a10, p.a13, COUNT(*) AS applications,
       ROUND(100.0 * SUM(f.is_positive) / COUNT(*), 1) AS positive_pct
{APP_JOIN}
GROUP BY p.a9, p.a10, p.a13
HAVING COUNT(*) >= 15
ORDER BY positive_pct DESC, applications DESC
LIMIT 8;"""),
 ("Q3", "Slice", "Slice on A9 = 't': how does the outcome vary across the 14 values of A6?",
  f"""SELECT a6.a6_value AS a6, COUNT(*) AS applications,
       ROUND(100.0 * SUM(f.is_positive) / COUNT(*), 1) AS positive_pct
FROM fact_application f
JOIN dim_profile p ON f.profile_key = p.profile_key
JOIN dim_a6 a6 ON f.a6_key = a6.a6_key
WHERE p.a9 = 't'
GROUP BY a6.a6_value
HAVING COUNT(*) >= 10
ORDER BY applications DESC;"""),
 ("Q4", "Dice", "Dice on A9 = 't', A2 band in (Mid, High) and A15 band in (Low, High): outcome by A4.",
  """SELECT p.a4, COUNT(*) AS applications, ROUND(100.0 * SUM(f.is_positive) / COUNT(*), 1) AS positive_pct,
       ROUND(AVG(f.a3), 2) AS mean_a3
FROM fact_application f
JOIN dim_profile p ON f.profile_key = p.profile_key
JOIN dim_band b2 ON f.a2_band_key = b2.band_key
JOIN dim_band b15 ON f.a15_band_key = b15.band_key
WHERE p.a9 = 't' AND b2.band_label IN ('Mid', 'High') AND b15.band_label IN ('Low', 'High')
GROUP BY p.a4
ORDER BY applications DESC;"""),
 ("Q5", "Pivot", "Cross-tabulate A4 against the outcome (pivot of outcome across columns).",
  f"""SELECT p.a4,
       SUM(CASE WHEN o.outcome_label = '+' THEN 1 ELSE 0 END) AS class_plus,
       SUM(CASE WHEN o.outcome_label = '-' THEN 1 ELSE 0 END) AS class_minus,
       ROUND(100.0 * SUM(CASE WHEN o.outcome_label = '+' THEN 1 ELSE 0 END) / COUNT(*), 1) AS plus_pct
{APP_JOIN}
GROUP BY p.a4
ORDER BY p.a4;"""),
 ("Q6", "Cross-tab of two bands", "How do the A11 band and the A15 band jointly relate to the outcome (cube-style view)?",
  """SELECT b11.band_label AS a11_band, b15.band_label AS a15_band, COUNT(*) AS applications,
       ROUND(100.0 * SUM(f.is_positive) / COUNT(*), 1) AS positive_pct
FROM fact_application f
JOIN dim_band b11 ON f.a11_band_key = b11.band_key
JOIN dim_band b15 ON f.a15_band_key = b15.band_key
GROUP BY b11.band_label, b15.band_label
HAVING COUNT(*) >= 20
ORDER BY b11.band_label, b15.band_label;"""),
 ("Q7", "Data-quality slice", "Does the outcome mix or the screening-model accuracy change with missing data?",
  """SELECT q.quality_label, COUNT(*) AS applications, ROUND(100.0 * AVG(f.is_positive), 1) AS positive_pct,
       ROUND(100.0 * AVG(f.is_correct), 1) AS model_accuracy_pct
FROM fact_application f JOIN dim_quality q ON f.quality_key = q.quality_key
GROUP BY q.quality_label ORDER BY q.min_missing;"""),
 ("Q8", "Screening summary by cluster", "Per K-Means cluster: outcome mix and the errors made by the decision-tree screen (out-of-fold).",
  """SELECT c.cluster_label, COUNT(*) AS applications, ROUND(100.0 * AVG(f.is_positive), 1) AS positive_pct,
       ROUND(100.0 * AVG(f.is_correct), 1) AS accuracy_pct,
       SUM(CASE WHEN f.cv_pred_class = 1 AND f.is_positive = 0 THEN 1 ELSE 0 END) AS false_positives,
       SUM(CASE WHEN f.cv_pred_class = 0 AND f.is_positive = 1 THEN 1 ELSE 0 END) AS false_negatives
FROM fact_application f JOIN dim_cluster c ON f.cluster_key = c.cluster_key
GROUP BY c.cluster_label ORDER BY c.cluster_label;"""),
 ("Q9", "Roll-up of experimental runs", "Average cross-validated accuracy by model and scaling, with a model-level roll-up.",
  """SELECT m.model_name, pp.scaling, COUNT(*) AS runs, ROUND(100 * AVG(r.acc_mean), 1) AS avg_accuracy_pct
FROM fact_experiment_run r JOIN dim_model m ON r.model_key = m.model_key
JOIN dim_preprocessing pp ON r.prep_key = pp.prep_key
WHERE pp.reduction = 'none' AND pp.encoding IN ('onehot', 'native')
GROUP BY m.model_name, pp.scaling
UNION ALL
SELECT m.model_name, 'ALL scalings', COUNT(*), ROUND(100 * AVG(r.acc_mean), 1)
FROM fact_experiment_run r JOIN dim_model m ON r.model_key = m.model_key
JOIN dim_preprocessing pp ON r.prep_key = pp.prep_key
WHERE pp.reduction = 'none' AND pp.encoding IN ('onehot', 'native')
GROUP BY m.model_name
ORDER BY 1, 2;"""),
 ("Q10", "Drill-down of experimental runs", "KNN accuracy by missing-value strategy (rows) and scaling (columns), one-hot encoding.",
  """SELECT pp.missing_strategy,
       ROUND(100 * MAX(CASE WHEN pp.scaling = 'none' THEN r.acc_mean END), 1) AS raw,
       ROUND(100 * MAX(CASE WHEN pp.scaling = 'minmax' THEN r.acc_mean END), 1) AS minmax,
       ROUND(100 * MAX(CASE WHEN pp.scaling = 'zscore' THEN r.acc_mean END), 1) AS zscore
FROM fact_experiment_run r JOIN dim_model m ON r.model_key = m.model_key
JOIN dim_preprocessing pp ON r.prep_key = pp.prep_key
WHERE m.model_family = 'Instance-based' AND pp.encoding = 'onehot' AND pp.reduction = 'none'
GROUP BY pp.missing_strategy ORDER BY pp.missing_strategy;"""),
 ("Q11", "Feature-group analysis", "Average importance rank of each documented feature group under the three selection methods.",
  """SELECT g.group_name, sm.method_name, COUNT(*) AS features, ROUND(AVG(fi.rank_in_method), 1) AS avg_rank
FROM fact_feature_importance fi
JOIN dim_feature ft ON fi.feature_key = ft.feature_key
JOIN dim_feature_group g ON ft.group_key = g.group_key
JOIN dim_selection_method sm ON fi.method_key = sm.method_key
GROUP BY g.group_name, sm.method_name
ORDER BY sm.method_name, avg_rank;"""),
]
sql_out = []
for qid, op, question, sql in QUERIES:
    res = pd.read_sql_query(sql, con)
    res.to_csv(f"{TAB}/olap_{qid}.csv", index=False)
    save_table(f"olap_{qid}", res)
    sql_out.append(f"-- {qid} [{op}]: {question}\n{sql}\n")
open(f"{WH}/olap_queries.sql", "w").write("\n".join(sql_out))
REPORT["_queries"] = [{"id": q[0], "op": q[1], "question": q[2], "sql": q[3]} for q in QUERIES]
con.close()

# %% [markdown]
# ## 3. Schema diagrams

# %%
import matplotlib.patches as mpatches

def schema_figure(boxes, edges, fname, title, size=(8, 6.2)):
    """boxes: name -> (cx, cy, lines, kind) ; edges: list of (name_a, name_b)"""
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off"); ax.set_title(title, fontsize=11)
    for a_, b_ in edges:
        ax.plot([boxes[a_][0], boxes[b_][0]], [boxes[a_][1], boxes[b_][1]], color="#777777", lw=1, zorder=0)
    for name, (cx, cy, lines, kind) in boxes.items():
        w, h = 3.5 if kind == "fact" else 2.4, 0.55 + 0.27 * len(lines)
        fc, ec = ("#FDF1E3", "#DD8452") if kind == "fact" else ("#EAF2FB", "#4C72B0")
        ax.add_patch(mpatches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.04", fc=fc, ec=ec, lw=1.2))
        ax.text(cx, cy + h / 2 - 0.22, name, ha="center", va="center", fontsize=8, fontweight="bold")
        for i, ln in enumerate(lines):
            ax.text(cx, cy + h / 2 - 0.52 - i * 0.27, ln, ha="center", va="center", fontsize=6.3)
    plt.savefig(f"{FIG}/{fname}.png"); plt.close()

schema_figure({
    "fact_application": (5, 4, ["PK app_id", "FK outcome_key, profile_key, a6_key, a7_key", "FK quality_key, cluster_key",
                                "FK a2/a3/a8/a11/a14/a15_band_key", "Measures: a2, a3, a8, a11, a14, a15",
                                "missing_count, is_positive", "cv_pred_prob, cv_pred_class, is_correct"], "fact"),
    "dim_outcome": (1.5, 7.0, ["outcome_key", "outcome_label (+ / -)"], "dim"),
    "dim_quality": (5, 7.0, ["quality_key", "quality_label", "min_missing"], "dim"),
    "dim_cluster": (8.5, 7.0, ["cluster_key", "algorithm", "cluster_label"], "dim"),
    "dim_profile (junk)": (1.5, 4.0, ["profile_key", "a1, a4, a5, a9", "a10, a12, a13"], "dim"),
    "dim_band (6 roles)": (8.5, 4.0, ["band_key", "band_label"], "dim"),
    "dim_a6": (2.4, 1.0, ["a6_key", "a6_value"], "dim"),
    "dim_a7": (7.6, 1.0, ["a7_key", "a7_value"], "dim")},
    [("fact_application", k) for k in ["dim_outcome", "dim_quality", "dim_cluster", "dim_profile (junk)", "dim_band (6 roles)", "dim_a6", "dim_a7"]],
    "21_warehouse_application", "Schema 1: application characteristics and outcomes")
schema_figure({
    "fact_experiment_run": (5, 6.6, ["PK run_key", "FK model_key, prep_key", "Measures: n_rows, acc_mean, acc_std", "prec_mean, rec_mean, f1_mean, auc_mean, time_s"], "fact"),
    "dim_model": (1.4, 6.6, ["model_key", "model_name", "model_family"], "dim"),
    "dim_preprocessing": (8.6, 6.6, ["prep_key", "missing_strategy, encoding", "scaling, reduction, k_features"], "dim"),
    "fact_feature_importance": (5, 3.4, ["PK feature_key + method_key", "Measures: score, rank_in_method"], "fact"),
    "dim_feature": (1.4, 3.4, ["feature_key, feature_name", "group_key, n_distinct, n_missing"], "dim"),
    "dim_selection_method": (8.6, 3.4, ["method_key", "method_name"], "dim"),
    "dim_feature_group": (1.4, 1.2, ["group_key, group_name", "description"], "dim")},
    [("fact_experiment_run", "dim_model"), ("fact_experiment_run", "dim_preprocessing"),
     ("fact_feature_importance", "dim_feature"), ("fact_feature_importance", "dim_selection_method"),
     ("dim_feature", "dim_feature_group")],
    "22_warehouse_experiments", "Schema 2: experimental runs and feature importance", size=(8, 5.6))
print("Part 6 done")
# %% [markdown]
# # Part 7: export all tables and scalar results for the report

# %%
def _clean(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

json.dump({"tables": REPORT, "num": NUM}, open("results.json", "w"), default=_clean, indent=1)
print("All parts finished; results.json written")
