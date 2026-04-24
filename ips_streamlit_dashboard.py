# ============================================================
# Streamlit Dashboard - IPS Portfolio 75-79 / Afore XXI Banorte
# Based on IPS_bueno.html workflow
# ============================================================

import io
import re
import unicodedata
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except Exception:
    yf = None

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="IPS Portfolio Dashboard 75-79",
    page_icon="📊",
    layout="wide",
)

st.title("IPS Portfolio Dashboard — SIEFORE Básica Generacional 75–79")
st.caption("Afore XXI Banorte | MXN 50 billion portfolio | Vector Analítico + Yahoo Finance proxies")

# -----------------------------
# Constants
# -----------------------------
TOTAL_PORTFOLIO_MXN = 50_000_000_000
TRADING_DAYS = 252
DEFAULT_RISK_FREE_RATE = 0.05

IPS_WEIGHTS = {
    "Mexican government fixed income": 0.45,
    "Domestic corporate and bank fixed income": 0.12,
    "International fixed income": 0.03,
    "Mexican equities": 0.08,
    "International equities": 0.15,
    "Structured instruments": 0.10,
    "FIBRAs": 0.03,
    "Commodities": 0.02,
    "Cash and liquid assets": 0.02,
}

SLEEVE_PROXY_CANDIDATES = {
    "Mexican government fixed income": ["CETETRC.MX", "M10TRACISHRS.MX", "UDITRACISHRS.MX", "SHV"],
    "Domestic corporate and bank fixed income": ["CORPTRCISHRS.MX", "LQD", "AGG"],
    "International fixed income": ["AGG", "BND", "EMB"],
    "Mexican equities": ["EWW", "^MXX", "MEXTRAC09.MX"],
    "International equities": ["ACWI", "SPY", "QQQ"],
    "Structured instruments": ["AOR", "QQQ", "SPY"],
    "FIBRAs": ["FUNO11.MX", "FMTY14.MX", "VNQ"],
    "Commodities": ["GLD", "SLV", "USO"],
    "Cash and liquid assets": ["SHV", "BIL", "SGOV"],
}

PROXY_EXPLANATION = {
    "Mexican government fixed income": "Proxy for Mexican sovereign fixed income because individual CETES/BONOS/UDIBONOS/IPABONOS have limited direct Yahoo tickers.",
    "Domestic corporate and bank fixed income": "Proxy for domestic corporate/bank debt; fallback uses investment-grade credit ETFs when local tickers are unavailable.",
    "International fixed income": "Proxy for global/U.S. aggregate and emerging-market USD bonds.",
    "Mexican equities": "Proxy for Mexican equity market exposure.",
    "International equities": "Proxy for broad developed/global equity exposure.",
    "Structured instruments": "Proxy for multi-asset/option-like structured payoff exposure using liquid ETFs.",
    "FIBRAs": "Proxy for Mexican real estate trust exposure.",
    "Commodities": "Proxy for liquid commodity exposures.",
    "Cash and liquid assets": "Proxy for short-duration T-bills/cash equivalents.",
}

# -----------------------------
# Helpers
# -----------------------------
def clean_txt(x):
    x = "" if pd.isna(x) else str(x)
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    x = x.upper().strip()
    return re.sub(r"\s+", " ", x)


def clean_key(x):
    return re.sub(r"[^A-Z0-9]", "", clean_txt(x))


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def normalize_series(s, higher_is_better=True):
    s = safe_numeric(s)
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=s.index)
    s = s.fillna(s.median())
    min_v, max_v = s.min(), s.max()
    if max_v == min_v:
        out = pd.Series(0.5, index=s.index)
    else:
        out = (s - min_v) / (max_v - min_v)
    return out if higher_is_better else 1 - out


def score_to_weights(score, total_weight, min_w=None, max_w=None):
    score = safe_numeric(score).fillna(0)
    n = len(score)
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([total_weight], index=score.index)

    score_std = score.std()
    z = (score - score.mean()) / (score_std if score_std and score_std > 0 else 1)
    raw = np.exp(z)
    weights = pd.Series(raw / raw.sum() * total_weight, index=score.index)

    if min_w is not None or max_w is not None:
        min_w = 0 if min_w is None else min_w
        max_w = total_weight if max_w is None else max_w
        for _ in range(30):
            weights = weights.clip(lower=min_w, upper=max_w)
            diff = total_weight - weights.sum()
            free = (weights > min_w + 1e-12) & (weights < max_w - 1e-12)
            if abs(diff) < 1e-12 or free.sum() == 0:
                break
            weights.loc[free] += diff * (weights.loc[free] / weights.loc[free].sum())
        weights = weights / weights.sum() * total_weight
    return weights


def annualized_return(daily_returns):
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan
    return (1 + daily_returns).prod() ** (TRADING_DAYS / len(daily_returns)) - 1


def annualized_volatility(daily_returns):
    daily_returns = daily_returns.dropna()
    return daily_returns.std() * np.sqrt(TRADING_DAYS) if len(daily_returns) > 0 else np.nan


def historical_var(daily_returns, confidence=0.95):
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return np.nan
    return -np.percentile(daily_returns, (1 - confidence) * 100)


def max_drawdown(cum_returns):
    if cum_returns.dropna().empty:
        return np.nan
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    return drawdown.min()


def format_pct(x):
    return "—" if pd.isna(x) else f"{x:.2%}"


def format_mxn(x):
    return "—" if pd.isna(x) else f"${x:,.0f}"

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data(show_spinner=False)
def load_vector_from_bytes(file_bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_selected = None
    header_row = None

    for sh in xls.sheet_names:
        preview = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sh, header=None, nrows=50)
        for i in range(len(preview)):
            row_clean = [clean_key(v) for v in preview.iloc[i].tolist()]
            if any("ISIN" in v for v in row_clean):
                sheet_selected = sh
                header_row = i
                break
        if sheet_selected is not None:
            break

    if sheet_selected is None:
        raise ValueError("No ISIN-like header was found in the first rows of the Vector file.")

    vector = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_selected, header=header_row)
    vector.columns = [str(c).strip() for c in vector.columns]

    rename = {}
    for c in vector.columns:
        ck = clean_key(c)
        if "ISIN" in ck:
            rename[c] = "ISIN"
        elif ck in ["TIPOVALOR", "TIPO", "TV", "CLASE", "INSTRUMENTO"]:
            rename[c] = "Tipo"
        elif "EMISORA" in ck or "EMISOR" in ck:
            rename[c] = "Emisora"
        elif "SERIE" in ck:
            rename[c] = "Serie"
        elif "CUPON" in ck:
            rename[c] = "Coupon"
        elif "TASA" in ck or "RENDIMIENTO" in ck or "YIELD" in ck:
            rename[c] = "Yield"
        elif "VENC" in ck or "MATURITY" in ck or "FECHAVTO" in ck or "FECHAVENC" in ck:
            rename[c] = "Maturity"
        elif "DURACION" in ck or "DURATION" in ck:
            rename[c] = "Duration"
        elif "PRECIO" in ck or "PRICE" in ck:
            rename[c] = "Price"
        elif "CALIFIC" in ck or "RATING" in ck:
            rename[c] = "Rating"
        elif "SECTOR" in ck:
            rename[c] = "Sector"

    vector = vector.rename(columns=rename)
    vector = vector.loc[:, ~vector.columns.duplicated()].copy()

    if "ISIN" not in vector.columns:
        raise ValueError(f"No ISIN column was found. Columns detected: {vector.columns.tolist()}")

    vector["ISIN"] = vector["ISIN"].astype(str).str.strip().str.upper()
    vector = vector[(vector["ISIN"].notna()) & (vector["ISIN"] != "") & (vector["ISIN"] != "NAN")].copy()

    for needed_col in ["Tipo", "Emisora", "Serie", "Yield", "Coupon", "Maturity", "Duration", "Price", "Rating", "Sector"]:
        if needed_col not in vector.columns:
            vector[needed_col] = np.nan

    for col in ["Yield", "Coupon", "Duration", "Price"]:
        if isinstance(vector[col], pd.DataFrame):
            vector[col] = vector[col].iloc[:, 0]
        vector[col] = pd.to_numeric(vector[col], errors="coerce")

    vector["Maturity"] = pd.to_datetime(vector["Maturity"], errors="coerce")
    return vector, sheet_selected, header_row

# -----------------------------
# Portfolio construction
# -----------------------------
def build_government_assets(vector):
    mandatory_gov = pd.DataFrame([
        ["MXBIGO000YV0", "CETES (91)",  "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "BI", "09/07/2026", 0.23, "CETES"],
        ["MXBIGO000YR8", "CETES (182)", "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "BI", "03/09/2026", 0.39, "CETES"],
        ["MXBIGO000YW8", "CETES (364)", "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "BI", "01/04/2027", 0.97, "CETES"],
        ["MXBIGO000YP2", "CETES (721)", "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "BI", "17/02/2028", 1.87, "CETES"],
        ["MXIMBP060209", "IPABONO",     "INSTITUTO PARA LA PROTECCION AL AHORRO BANCARIO", "IM", "01/02/2029", 2.58, "IPAB"],
        ["MXIQBP0701X0", "IPABONO",     "INSTITUTO PARA LA PROTECCION AL AHORRO BANCARIO", "IQ", "16/01/2031", 4.08, "IPAB"],
        ["MXISBP0401R5", "IPABONO",     "INSTITUTO PARA LA PROTECCION AL AHORRO BANCARIO", "IS", "07/04/2033", 5.69, "IPAB"],
        ["MXLDGO000579", "BONDES",      "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "LD", "06/08/2026", 0.31, "BONDES"],
        ["MXLFGO0003M3", "BONDES",      "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "LF", "19/04/2035", 6.80, "BONDES"],
        ["MXLGGO0000C8", "BONDES",      "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "LG", "24/07/2031", 4.49, "BONDES"],
        ["MX0SGO0000P9", "UDIBONOS",    "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "S",  "24/08/2034", 7.21, "UDIBONOS"],
        ["MX0SGO000098", "UDIBONOS",    "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "S",  "15/11/2040", 11.05, "UDIBONOS"],
        ["MX0MGO0000J5", "BONOSM",      "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "M",  "18/11/2038", 7.51, "BONOS M 10-20 años"],
        ["MXMSGO000001", "BONOSM",      "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "MS", "24/05/2035", 6.41, "BONOS M 10-20 años"],
        ["MX0MGO000102", "BONOSM",      "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "M",  "07/11/2047", 9.22, "BONOS M 20-30 años"],
        ["MX0MGO0001E4", "BONOSM",      "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "M",  "31/07/2053", 9.94, "BONOS M 20-30 años"],
        ["US91086QAV05", "UMS",         "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "",   "11/01/2040", np.nan, "UMS"],
        ["JP548400DG68", "UMS",         "SECRETARIA DE HACIENDA Y CREDITO PUBLICO", "",   "16/06/2036", np.nan, "UMS"],
    ], columns=["ISIN", "Tipo", "Emisora", "Serie", "Maturity", "Duration", "Sovereign Bucket"])

    sovereign_bucket_weights = {
        "BONOS M 10-20 años": 0.30,
        "BONOS M 20-30 años": 0.15,
        "UDIBONOS": 0.25,
        "BONDES": 0.12,
        "IPAB": 0.08,
        "CETES": 0.05,
        "UMS": 0.05,
    }

    gov = mandatory_gov.copy()
    gov["Maturity"] = pd.to_datetime(gov["Maturity"], dayfirst=True, errors="coerce")

    vector_fin_cols = [c for c in ["ISIN", "Yield", "Coupon", "Price", "Rating", "Sector"] if c in vector.columns]
    if len(vector_fin_cols) > 1:
        gov = gov.merge(vector[vector_fin_cols].drop_duplicates(subset=["ISIN"]), on="ISIN", how="left")

    fallback_gov_yields = {
        "CETES": 0.090,
        "IPAB": 0.092,
        "BONDES": 0.091,
        "UDIBONOS": 0.045,
        "BONOS M 10-20 años": 0.094,
        "BONOS M 20-30 años": 0.096,
        "UMS": 0.055,
    }

    gov["Yield Used"] = safe_numeric(gov["Yield"] if "Yield" in gov.columns else np.nan)
    gov.loc[gov["Yield Used"] > 1, "Yield Used"] = gov.loc[gov["Yield Used"] > 1, "Yield Used"] / 100
    gov["Yield Used"] = gov.apply(lambda r: r["Yield Used"] if pd.notna(r["Yield Used"]) else fallback_gov_yields.get(r["Sovereign Bucket"], np.nan), axis=1)

    gov["Yield Score"] = gov.groupby("Sovereign Bucket")["Yield Used"].transform(lambda x: normalize_series(x, True))
    gov["Duration Score"] = gov.groupby("Sovereign Bucket")["Duration"].transform(lambda x: normalize_series(x, False))
    gov["Maturity Days"] = (gov["Maturity"] - pd.Timestamp.today().normalize()).dt.days
    gov["Liquidity Score"] = gov.groupby("Sovereign Bucket")["Maturity Days"].transform(lambda x: normalize_series(x, False))
    gov["Financial Score"] = 0.55 * gov["Yield Score"] + 0.35 * gov["Duration Score"] + 0.10 * gov["Liquidity Score"]
    gov["Sovereign Bucket Weight"] = gov["Sovereign Bucket"].map(sovereign_bucket_weights)

    gov["Weight within Sovereign Bucket"] = 0.0
    for bucket, target_weight in sovereign_bucket_weights.items():
        idx = gov.index[gov["Sovereign Bucket"] == bucket]
        if len(idx) == 0:
            continue
        n = len(idx)
        min_w = 0.15 * target_weight / n if n > 1 else target_weight
        max_w = 0.65 * target_weight if n > 1 else target_weight
        gov.loc[idx, "Weight within Sovereign Bucket"] = score_to_weights(gov.loc[idx, "Financial Score"], target_weight, min_w, max_w)

    gov["Weight"] = IPS_WEIGHTS["Mexican government fixed income"] * gov["Weight within Sovereign Bucket"]
    gov["Sleeve"] = "Mexican government fixed income"
    gov["Asset Name"] = gov["Tipo"] + " " + gov["Serie"].fillna("").astype(str)
    gov["Source"] = "Mandatory IPS government list; optimized inside fixed sovereign buckets"
    gov["Yahoo Proxy"] = np.nan

    sovereign_validation = gov.groupby("Sovereign Bucket", as_index=False).agg(
        Assets=("ISIN", "count"),
        Weight_within_Sovereign_Bucket=("Weight within Sovereign Bucket", "sum"),
        Portfolio_Weight=("Weight", "sum"),
    )
    sovereign_validation["Target within Sovereign Bucket"] = sovereign_validation["Sovereign Bucket"].map(sovereign_bucket_weights)
    sovereign_validation["Difference"] = sovereign_validation["Weight_within_Sovereign_Bucket"] - sovereign_validation["Target within Sovereign Bucket"]

    return gov, sovereign_validation, mandatory_gov


def select_corporate_assets(vector, mandatory_gov, n_corporate):
    gov_keywords = [
        "SECRETARIA DE HACIENDA", "GOBIERNO", "BANCO DE MEXICO", "BANXICO",
        "INSTITUTO PARA LA PROTECCION", "IPAB", "TESORERIA", "ESTADOS UNIDOS MEXICANOS"
    ]

    v = vector.copy()
    v["Emisora_clean"] = v["Emisora"].astype(str).str.upper()
    is_gov_emitter = v["Emisora_clean"].apply(lambda x: any(k in x for k in gov_keywords))
    is_mandatory_gov = v["ISIN"].isin(mandatory_gov["ISIN"])
    corp_candidates = v[~is_gov_emitter & ~is_mandatory_gov].drop_duplicates(subset=["ISIN"]).copy()

    for c in ["Yield", "Coupon", "Duration", "Price"]:
        corp_candidates[c] = safe_numeric(corp_candidates[c])

    corp_candidates.loc[corp_candidates["Yield"] > 1, "Yield"] = corp_candidates.loc[corp_candidates["Yield"] > 1, "Yield"] / 100

    corp_candidates["data_score"] = 0
    corp_candidates["data_score"] += corp_candidates["Yield"].notna().astype(int) * 3
    corp_candidates["data_score"] += corp_candidates["Coupon"].notna().astype(int) * 2
    corp_candidates["data_score"] += corp_candidates["Maturity"].notna().astype(int) * 2
    corp_candidates["data_score"] += corp_candidates["Duration"].notna().astype(int)
    corp_candidates["Price_Sanity"] = corp_candidates["Price"].between(80, 120).astype(int)
    corp_candidates["data_score"] += corp_candidates["Price_Sanity"]

    corp_candidates["Yield Score"] = normalize_series(corp_candidates["Yield"], True)
    corp_candidates["Coupon Score"] = normalize_series(corp_candidates["Coupon"], True)
    corp_candidates["Duration Score"] = normalize_series(corp_candidates["Duration"], False)
    corp_candidates["Data Quality Score"] = normalize_series(corp_candidates["data_score"], True)
    corp_candidates["Selection Score"] = (
        0.45 * corp_candidates["Yield Score"]
        + 0.20 * corp_candidates["Coupon Score"]
        + 0.20 * corp_candidates["Duration Score"]
        + 0.15 * corp_candidates["Data Quality Score"]
    )

    corp_candidates = corp_candidates.sort_values(["Selection Score", "Yield"], ascending=[False, False])
    corp_selected = corp_candidates.groupby("Emisora", dropna=False).head(2)
    if len(corp_selected) < n_corporate:
        needed = n_corporate - len(corp_selected)
        remaining = corp_candidates[~corp_candidates["ISIN"].isin(corp_selected["ISIN"])]
        corp_selected = pd.concat([corp_selected, remaining.head(needed)], ignore_index=True)

    corp_selected = corp_selected.head(n_corporate).copy()
    corp_selected["Sleeve"] = "Domestic corporate and bank fixed income"
    corp_selected["Asset Name"] = corp_selected["Emisora"].astype(str) + " " + corp_selected["Serie"].astype(str)
    corp_selected["Source"] = "Vector Analítico selection; optimized by yield/risk/data quality"
    corp_selected["Yahoo Proxy"] = np.nan

    n_corp = max(len(corp_selected), 1)
    min_asset_weight = IPS_WEIGHTS["Domestic corporate and bank fixed income"] * 0.35 / n_corp
    max_asset_weight = IPS_WEIGHTS["Domestic corporate and bank fixed income"] * 0.075

    corp_selected["Weight"] = score_to_weights(
        corp_selected["Selection Score"],
        total_weight=IPS_WEIGHTS["Domestic corporate and bank fixed income"],
        min_w=min_asset_weight,
        max_w=max_asset_weight,
    ).values

    issuer_cap = IPS_WEIGHTS["Domestic corporate and bank fixed income"] * 0.15
    for _ in range(10):
        issuer_weights = corp_selected.groupby("Emisora")["Weight"].sum()
        over_issuers = issuer_weights[issuer_weights > issuer_cap]
        if over_issuers.empty:
            break
        excess_total = 0
        for issuer, total_w in over_issuers.items():
            mask = corp_selected["Emisora"] == issuer
            scale = issuer_cap / total_w
            old = corp_selected.loc[mask, "Weight"].copy()
            corp_selected.loc[mask, "Weight"] = old * scale
            excess_total += (old - corp_selected.loc[mask, "Weight"]).sum()
        under_mask = ~corp_selected["Emisora"].isin(over_issuers.index)
        if under_mask.sum() > 0 and excess_total > 0:
            corp_selected.loc[under_mask, "Weight"] += (corp_selected.loc[under_mask, "Weight"] / corp_selected.loc[under_mask, "Weight"].sum()) * excess_total

    corp_selected["Weight"] = corp_selected["Weight"] / corp_selected["Weight"].sum() * IPS_WEIGHTS["Domestic corporate and bank fixed income"]
    return corp_selected, corp_candidates


@st.cache_data(show_spinner=False)
def download_proxy_prices_cached(tickers, period="3y", start=None):
    if yf is None:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers=tickers,
            period=period if start is None else None,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if data is None or len(data) == 0:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                prices = data["Close"].copy()
            elif "Adj Close" in data.columns.get_level_values(0):
                prices = data["Adj Close"].copy()
            else:
                prices = data.xs(data.columns.get_level_values(0)[0], axis=1, level=0).copy()
        else:
            if "Close" in data.columns:
                prices = data[["Close"]].rename(columns={"Close": tickers[0] if isinstance(tickers, list) else tickers})
            else:
                prices = data.copy()
        return prices.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def build_other_assets(period="3y"):
    other_assets = pd.DataFrame([
        ["International fixed income", "iShares Core U.S. Aggregate Bond ETF", "US4642872265", "AGG"],
        ["International fixed income", "iShares J.P. Morgan USD Emerging Markets Bond ETF", "US4642882819", "EMB"],
        ["Mexican equities", "iShares MSCI Mexico ETF", "US4642868222", "EWW"],
        ["Mexican equities", "Mexico Equity Proxy / IPC", "", "^MXX"],
        ["International equities", "SPDR S&P 500 ETF Trust", "US78462F1030", "SPY"],
        ["International equities", "iShares MSCI ACWI ETF", "US4642882579", "ACWI"],
        ["International equities", "Invesco QQQ Trust", "US46090E1038", "QQQ"],
        ["Structured instruments", "Global multi-asset structured proxy", "", "AOR"],
        ["Structured instruments", "NASDAQ structured payoff proxy", "US46090E1038", "QQQ"],
        ["FIBRAs", "FIBRA Uno", "MXCFFU000001", "FUNO11.MX"],
        ["FIBRAs", "FIBRA Monterrey", "MXCFFM010003", "FMTY14.MX"],
        ["Commodities", "SPDR Gold Shares", "US78463V1070", "GLD"],
        ["Commodities", "iShares Silver Trust", "US46428Q1094", "SLV"],
        ["Commodities", "United States Oil Fund", "US91232N2071", "USO"],
        ["Cash and liquid assets", "iShares Short Treasury Bond ETF", "US4642886794", "SHV"],
        ["Cash and liquid assets", "SPDR Bloomberg 1-3 Month T-Bill ETF", "US78468R6633", "BIL"],
    ], columns=["Sleeve", "Asset Name", "ISIN", "Yahoo Proxy"])

    fallback_proxy_score = {
        "AGG": 0.55, "EMB": 0.45,
        "EWW": 0.60, "^MXX": 0.40,
        "SPY": 0.45, "ACWI": 0.35, "QQQ": 0.20,
        "AOR": 0.55,
        "FUNO11.MX": 0.55, "FMTY14.MX": 0.45,
        "GLD": 0.50, "SLV": 0.25, "USO": 0.25,
        "SHV": 0.70, "BIL": 0.30,
    }

    tickers_other = other_assets["Yahoo Proxy"].dropna().unique().tolist()
    proxy_prices_pre = download_proxy_prices_cached(tickers_other, period=period)
    proxy_returns_pre = proxy_prices_pre.pct_change().dropna(how="all") if not proxy_prices_pre.empty else pd.DataFrame()

    proxy_perf = []
    for ticker in tickers_other:
        if ticker in proxy_returns_pre.columns and proxy_returns_pre[ticker].dropna().shape[0] > 60:
            r = proxy_returns_pre[ticker].dropna()
            ann_ret = annualized_return(r)
            ann_vol = annualized_volatility(r)
            sharpe_like = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
            proxy_perf.append([ticker, ann_ret, ann_vol, sharpe_like])
        else:
            proxy_perf.append([ticker, np.nan, np.nan, np.nan])

    proxy_perf = pd.DataFrame(proxy_perf, columns=["Yahoo Proxy", "Proxy Annual Return", "Proxy Annual Volatility", "Proxy Sharpe Like"])
    other_assets = other_assets.merge(proxy_perf, on="Yahoo Proxy", how="left")
    other_assets["Fallback Score"] = other_assets["Yahoo Proxy"].map(fallback_proxy_score).fillna(0.50)

    other_assets["Proxy Return Score"] = other_assets.groupby("Sleeve")["Proxy Annual Return"].transform(lambda x: normalize_series(x, True))
    other_assets["Proxy Risk Score"] = other_assets.groupby("Sleeve")["Proxy Annual Volatility"].transform(lambda x: normalize_series(x, False))
    other_assets["Proxy Sharpe Score"] = other_assets.groupby("Sleeve")["Proxy Sharpe Like"].transform(lambda x: normalize_series(x, True))
    other_assets["Fallback Score Norm"] = other_assets.groupby("Sleeve")["Fallback Score"].transform(lambda x: normalize_series(x, True))

    has_yahoo = other_assets["Proxy Sharpe Like"].notna()
    other_assets["Optimization Score"] = np.where(
        has_yahoo,
        0.50 * other_assets["Proxy Sharpe Score"] + 0.30 * other_assets["Proxy Return Score"] + 0.20 * other_assets["Proxy Risk Score"],
        other_assets["Fallback Score Norm"],
    )

    other_assets["Weight"] = 0.0
    other_assets["Sleeve Internal Weight"] = 0.0
    for sleeve, target_w in IPS_WEIGHTS.items():
        idx = other_assets.index[other_assets["Sleeve"] == sleeve]
        if len(idx) == 0:
            continue
        n = len(idx)
        min_w = target_w * 0.20 / n if n > 1 else target_w
        max_w = target_w * 0.75 if n > 1 else target_w
        w = score_to_weights(other_assets.loc[idx, "Optimization Score"], target_w, min_w, max_w)
        other_assets.loc[idx, "Weight"] = w.values
        other_assets.loc[idx, "Sleeve Internal Weight"] = other_assets.loc[idx, "Weight"] / target_w

    other_assets["Tipo"] = "ETF / Proxy"
    other_assets["Emisora"] = other_assets["Asset Name"]
    other_assets["Serie"] = ""
    other_assets["Yield"] = np.nan
    other_assets["Coupon"] = np.nan
    other_assets["Maturity"] = pd.NaT
    other_assets["Duration"] = np.nan
    other_assets["Price"] = np.nan
    other_assets["Rating"] = np.nan
    other_assets["Sector"] = np.nan
    other_assets["Source"] = "Yahoo Finance proxy; optimized by historical return/risk when available"
    return other_assets


def consolidate_portfolio(gov, corp_selected, other_assets):
    common_cols = [
        "Sleeve", "Asset Name", "ISIN", "Tipo", "Emisora", "Serie",
        "Yield", "Yield Used", "Coupon", "Maturity", "Duration", "Price", "Rating", "Sector",
        "Financial Score", "Selection Score", "Optimization Score",
        "Weight", "Yahoo Proxy", "Source",
    ]

    gov_assets = gov.copy()
    corp_assets = corp_selected.copy()
    dfs = []
    for df in [gov_assets, corp_assets, other_assets.copy()]:
        for c in common_cols:
            if c not in df.columns:
                df[c] = np.nan
        dfs.append(df[common_cols].copy())

    portfolio_assets = pd.concat(dfs, ignore_index=True)
    portfolio_assets["Weight"] = safe_numeric(portfolio_assets["Weight"]).fillna(0)
    portfolio_assets["Weight %"] = portfolio_assets["Weight"] * 100
    portfolio_assets["Amount MXN"] = portfolio_assets["Weight"] * TOTAL_PORTFOLIO_MXN

    fallback_yields = {
        "Mexican government fixed income": 0.095,
        "Domestic corporate and bank fixed income": 0.105,
        "International fixed income": 0.045,
        "Mexican equities": 0.085,
        "International equities": 0.075,
        "Structured instruments": 0.080,
        "FIBRAs": 0.075,
        "Commodities": 0.030,
        "Cash and liquid assets": 0.050,
    }

    portfolio_assets["Yield Used"] = portfolio_assets.apply(
        lambda r: r["Yield Used"] if pd.notna(r["Yield Used"])
        else (r["Yield"] if pd.notna(r["Yield"]) else fallback_yields.get(r["Sleeve"], np.nan)),
        axis=1,
    )
    portfolio_assets["Yield Used"] = safe_numeric(portfolio_assets["Yield Used"])
    portfolio_assets.loc[portfolio_assets["Yield Used"] > 1, "Yield Used"] = portfolio_assets.loc[portfolio_assets["Yield Used"] > 1, "Yield Used"] / 100

    allocation_rows = []
    for sleeve, temp in portfolio_assets.groupby("Sleeve"):
        w = temp["Weight"].fillna(0)
        total_w = w.sum()
        allocation_rows.append({
            "Sleeve": sleeve,
            "Weight": total_w,
            "Amount_MXN": temp["Amount MXN"].sum(),
            "Assets": len(temp),
            "Weighted_Yield": np.average(temp["Yield Used"].fillna(0), weights=w) if total_w > 0 else np.nan,
            "Weighted_Duration": np.average(temp["Duration"].fillna(temp["Duration"].median()), weights=w) if total_w > 0 and temp["Duration"].notna().any() else np.nan,
        })

    allocation_check = pd.DataFrame(allocation_rows)
    allocation_check["Weight %"] = allocation_check["Weight"] * 100
    allocation_check["Target Weight"] = allocation_check["Sleeve"].map(IPS_WEIGHTS)
    allocation_check["Difference"] = allocation_check["Weight"] - allocation_check["Target Weight"]

    portfolio_assets = portfolio_assets.sort_values(["Sleeve", "Weight"], ascending=[True, False]).reset_index(drop=True)
    return portfolio_assets, allocation_check


def build_proxy_table():
    return pd.DataFrame([
        {
            "Sleeve": sleeve,
            "Primary Proxy": candidates[0],
            "Alternative Proxies": ", ".join(candidates[1:]),
            "Explanation": PROXY_EXPLANATION.get(sleeve, ""),
        }
        for sleeve, candidates in SLEEVE_PROXY_CANDIDATES.items()
    ])


def run_backtest(backtest_start, risk_free_rate):
    all_candidate_tickers = sorted(set(sum(SLEEVE_PROXY_CANDIDATES.values(), [])))
    prices_all = download_proxy_prices_cached(all_candidate_tickers, start=backtest_start)
    prices_all = prices_all.dropna(axis=1, how="all").ffill().dropna(how="all")

    selected_sleeve_proxy = {}
    for sleeve, candidates in SLEEVE_PROXY_CANDIDATES.items():
        selected = None
        for t in candidates:
            if t in prices_all.columns and prices_all[t].dropna().shape[0] > 60:
                selected = t
                break
        selected_sleeve_proxy[sleeve] = selected

    selected_proxy_table = pd.DataFrame([
        {"Sleeve": s, "Selected Proxy": p, "Status": "OK" if p else "NO DATA"}
        for s, p in selected_sleeve_proxy.items()
    ])

    sleeve_returns = pd.DataFrame(index=prices_all.index)
    for sleeve, ticker in selected_sleeve_proxy.items():
        if ticker is not None and ticker in prices_all.columns:
            sleeve_returns[sleeve] = prices_all[ticker].pct_change()

    sleeve_returns = sleeve_returns.dropna(how="all").fillna(0)
    if sleeve_returns.empty:
        return prices_all, selected_proxy_table, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float), None, pd.DataFrame(), pd.DataFrame()

    sleeve_weights = pd.Series(IPS_WEIGHTS)
    sleeve_weights = sleeve_weights[sleeve_weights.index.isin(sleeve_returns.columns)]
    sleeve_weights = sleeve_weights / sleeve_weights.sum()

    portfolio_returns = sleeve_returns.mul(sleeve_weights, axis=1).sum(axis=1)
    portfolio_cum = (1 + portfolio_returns).cumprod()

    benchmark_candidates = ["ACWI", "SPY", "EWW", "^MXX"]
    benchmark_ticker = next((t for t in benchmark_candidates if t in prices_all.columns), None)

    if benchmark_ticker:
        benchmark_returns = prices_all[benchmark_ticker].pct_change().reindex(portfolio_returns.index).fillna(0)
        benchmark_cum = (1 + benchmark_returns).cumprod()
    else:
        benchmark_returns = pd.Series(index=portfolio_returns.index, data=np.nan)
        benchmark_cum = pd.Series(index=portfolio_returns.index, data=np.nan)

    backtesting_table = pd.DataFrame({
        "Portfolio Daily Return": portfolio_returns.values,
        "Portfolio Cumulative Return Index": portfolio_cum.values,
        "Benchmark Ticker": benchmark_ticker,
        "Benchmark Daily Return": benchmark_returns.values,
        "Benchmark Cumulative Return Index": benchmark_cum.values,
    }, index=portfolio_returns.index)
    backtesting_table.index.name = "Date"

    return prices_all, selected_proxy_table, sleeve_returns, backtesting_table, portfolio_returns, portfolio_cum, benchmark_ticker, benchmark_returns, benchmark_cum


def calculate_metrics(portfolio_assets, portfolio_returns, portfolio_cum, benchmark_returns, benchmark_ticker, risk_free_rate):
    if portfolio_returns.empty:
        return pd.DataFrame(), np.nan, np.nan, np.nan, np.nan

    port_ann_return = annualized_return(portfolio_returns)
    port_ann_vol = annualized_volatility(portfolio_returns)
    port_var_95_daily = historical_var(portfolio_returns, 0.95)
    port_var_95_annual = port_var_95_daily * np.sqrt(TRADING_DAYS)
    port_sharpe = (port_ann_return - risk_free_rate) / port_ann_vol if port_ann_vol and port_ann_vol != 0 else np.nan
    port_yield = np.average(portfolio_assets["Yield Used"].fillna(0), weights=portfolio_assets["Weight"])
    port_mdd = max_drawdown(portfolio_cum)

    if benchmark_ticker and benchmark_returns.notna().sum() > 60:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]
        cov = np.cov(aligned["portfolio"], aligned["benchmark"])[0, 1]
        var_b = np.var(aligned["benchmark"])
        beta = cov / var_b if var_b != 0 else np.nan
        alpha_daily = aligned["portfolio"].mean() - beta * aligned["benchmark"].mean()
        alpha_annual = alpha_daily * TRADING_DAYS
    else:
        beta = np.nan
        alpha_annual = np.nan

    metrics = pd.DataFrame({
        "Metric": [
            "Expected / Backtested Annual Return",
            "Weighted Portfolio Yield",
            "Annualized Volatility",
            "Daily Historical VaR 95%",
            "Annualized Historical VaR 95%",
            "Daily VaR 95% in MXN",
            "Annualized VaR 95% in MXN",
            "Sharpe Ratio",
            "Alpha vs Benchmark",
            "Beta vs Benchmark",
            "Maximum Drawdown",
            "Benchmark",
        ],
        "Value": [
            port_ann_return,
            port_yield,
            port_ann_vol,
            port_var_95_daily,
            port_var_95_annual,
            port_var_95_daily * TOTAL_PORTFOLIO_MXN,
            port_var_95_annual * TOTAL_PORTFOLIO_MXN,
            port_sharpe,
            alpha_annual,
            beta,
            port_mdd,
            benchmark_ticker,
        ],
    })
    return metrics, port_ann_return, port_ann_vol, port_var_95_daily, port_sharpe

# -----------------------------
# Charts
# -----------------------------
def pie_allocation_chart(pie_data):
    df = pie_data.copy()
    df["Hover_Text"] = (
        "<b>" + df["Sleeve"].astype(str) + "</b><br>"
        + "Weight: " + df["Weight %"].round(2).astype(str) + "%<br>"
        + "Amount MXN: $" + df["Amount_MXN"].round(0).map("{:,.0f}".format) + "<br>"
        + "Assets: " + df["Assets"].astype(int).astype(str)
    )
    fig = go.Figure(data=[go.Pie(
        labels=df["Sleeve"],
        values=df["Weight"],
        hole=0.35,
        textinfo="percent+label",
        textposition="inside",
        hoverinfo="text",
        hovertext=df["Hover_Text"],
    )])
    fig.update_layout(title="Portfolio Allocation by Asset Type", height=620, legend_title_text="Asset Type")
    return fig


def bar_allocation_chart(pie_data):
    df = pie_data.copy()
    df["Amount MXN Billions"] = df["Amount_MXN"] / 1e9
    df["Hover_Text"] = (
        "<b>" + df["Sleeve"].astype(str) + "</b><br>"
        + "Amount: " + df["Amount MXN Billions"].round(2).astype(str) + " billion MXN<br>"
        + "Weight: " + df["Weight %"].round(2).astype(str) + "%<br>"
        + "Assets: " + df["Assets"].astype(int).astype(str)
    )
    fig = go.Figure(data=[go.Bar(
        x=df["Sleeve"],
        y=df["Amount MXN Billions"],
        text=df["Weight %"].round(1).astype(str) + "%",
        textposition="outside",
        hoverinfo="text",
        hovertext=df["Hover_Text"],
    )])
    fig.update_layout(title="Portfolio Allocation Amount by Sleeve", xaxis_title="Asset Type", yaxis_title="Amount MXN Billions", height=600)
    return fig


def sovereign_chart(sovereign_validation):
    df = sovereign_validation.copy()
    df["Hover_Text"] = (
        "<b>" + df["Sovereign Bucket"].astype(str) + "</b><br>"
        + "Weight inside sovereign bucket: " + (df["Weight_within_Sovereign_Bucket"] * 100).round(2).astype(str) + "%<br>"
        + "Portfolio weight: " + (df["Portfolio_Weight"] * 100).round(2).astype(str) + "%<br>"
        + "Number of assets: " + df["Assets"].astype(int).astype(str)
    )
    fig = go.Figure(data=[go.Pie(
        labels=df["Sovereign Bucket"],
        values=df["Weight_within_Sovereign_Bucket"],
        hole=0.35,
        textinfo="percent+label",
        textposition="inside",
        hoverinfo="text",
        hovertext=df["Hover_Text"],
    )])
    fig.update_layout(title="Sovereign Bucket Allocation inside Mexican Government Fixed Income", height=600, legend_title_text="Instrument")
    return fig


def backtest_chart(backtesting_table, benchmark_ticker):
    plot_bt = backtesting_table.reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_bt["Date"],
        y=plot_bt["Portfolio Cumulative Return Index"],
        mode="lines",
        name="Portfolio",
        hovertemplate="Date: %{x}<br>Portfolio Index: %{y:.3f}<extra></extra>",
    ))
    if benchmark_ticker:
        fig.add_trace(go.Scatter(
            x=plot_bt["Date"],
            y=plot_bt["Benchmark Cumulative Return Index"],
            mode="lines",
            name=f"Benchmark ({benchmark_ticker})",
            hovertemplate="Date: %{x}<br>Benchmark Index: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(title="Backtesting: Portfolio vs Benchmark", xaxis_title="Date", yaxis_title="Growth of $1", height=600, hovermode="x unified")
    return fig


def drawdown_chart(portfolio_cum):
    drawdown = portfolio_cum / portfolio_cum.cummax() - 1
    fig = go.Figure(data=[go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill="tozeroy",
        mode="lines",
        name="Drawdown",
        hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>",
    )])
    fig.update_layout(title="Portfolio Drawdown", yaxis_tickformat=".0%", xaxis_title="Date", yaxis_title="Drawdown", height=500)
    return fig


def returns_histogram(portfolio_returns, var_95_daily):
    fig = go.Figure(data=[go.Histogram(x=portfolio_returns.dropna(), nbinsx=60, name="Daily Returns")])
    if pd.notna(var_95_daily):
        fig.add_vline(x=-var_95_daily, line_dash="dash", annotation_text="VaR 95%", annotation_position="top left")
    fig.update_layout(title="Distribution of Portfolio Daily Returns", xaxis_tickformat=".1%", xaxis_title="Daily Return", yaxis_title="Frequency", height=550)
    return fig


def sleeve_risk_return_chart(sleeve_returns, selected_proxy_table):
    rows = []
    proxy_map = dict(zip(selected_proxy_table["Sleeve"], selected_proxy_table["Selected Proxy"]))
    for sleeve in sleeve_returns.columns:
        r = sleeve_returns[sleeve].dropna()
        rows.append({
            "Sleeve": sleeve,
            "Annual Return": annualized_return(r),
            "Annual Volatility": annualized_volatility(r),
            "Weight": IPS_WEIGHTS.get(sleeve, np.nan),
            "Selected Proxy": proxy_map.get(sleeve),
        })
    sleeve_stats = pd.DataFrame(rows)
    if sleeve_stats.empty:
        return go.Figure(), sleeve_stats
    sleeve_stats["Weight %"] = sleeve_stats["Weight"] * 100
    fig = go.Figure(data=[go.Scatter(
        x=sleeve_stats["Annual Volatility"],
        y=sleeve_stats["Annual Return"],
        mode="markers+text",
        text=sleeve_stats["Sleeve"],
        textposition="top center",
        marker=dict(size=np.maximum(sleeve_stats["Weight %"].fillna(1) * 2, 8)),
        hovertext=(
            "<b>" + sleeve_stats["Sleeve"].astype(str) + "</b><br>"
            + "Proxy: " + sleeve_stats["Selected Proxy"].astype(str) + "<br>"
            + "Weight: " + sleeve_stats["Weight %"].round(2).astype(str) + "%<br>"
            + "Annual Return: " + (sleeve_stats["Annual Return"] * 100).round(2).astype(str) + "%<br>"
            + "Annual Volatility: " + (sleeve_stats["Annual Volatility"] * 100).round(2).astype(str) + "%"
        ),
        hoverinfo="text",
    )])
    fig.update_layout(title="Risk / Return by Asset Sleeve Proxy", xaxis_tickformat=".1%", yaxis_tickformat=".1%", xaxis_title="Annualized Volatility", yaxis_title="Annualized Return", height=600)
    return fig, sleeve_stats

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Inputs")
uploaded_vector = st.sidebar.file_uploader("Upload VectorAnalitico20260416MD.xlsx", type=["xlsx"])
use_local = st.sidebar.checkbox("Use local VectorAnalitico20260416MD.xlsx if no upload", value=True)
n_corporate = st.sidebar.slider("Number of corporate/bank instruments", min_value=10, max_value=40, value=30, step=1)
backtest_start = st.sidebar.text_input("Backtest start date", value="2019-01-01")
risk_free_rate = st.sidebar.number_input("Risk-free rate assumption", min_value=0.0, max_value=0.30, value=DEFAULT_RISK_FREE_RATE, step=0.005, format="%.3f")

# Load vector bytes
file_bytes = None
if uploaded_vector is not None:
    file_bytes = uploaded_vector.getvalue()
elif use_local:
    try:
        with open("VectorAnalitico20260416MD.xlsx", "rb") as f:
            file_bytes = f.read()
    except FileNotFoundError:
        st.sidebar.warning("Local VectorAnalitico20260416MD.xlsx was not found. Upload the file to run the dashboard.")

if file_bytes is None:
    st.info("Upload the Vector Analítico Excel file in the sidebar to build the portfolio.")
    st.stop()

# -----------------------------
# Main pipeline
# -----------------------------
with st.spinner("Loading Vector and constructing portfolio..."):
    vector, sheet_selected, header_row = load_vector_from_bytes(file_bytes)
    gov, sovereign_validation, mandatory_gov = build_government_assets(vector)
    corp_selected, corp_candidates = select_corporate_assets(vector, mandatory_gov, n_corporate)
    other_assets = build_other_assets(period="3y")
    portfolio_assets, allocation_check = consolidate_portfolio(gov, corp_selected, other_assets)
    proxy_table = build_proxy_table()

with st.spinner("Downloading Yahoo Finance proxies and running backtest..."):
    prices_all, selected_proxy_table, sleeve_returns, backtesting_table, portfolio_returns, portfolio_cum, benchmark_ticker, benchmark_returns, benchmark_cum = run_backtest(backtest_start, risk_free_rate)
    metrics, port_ann_return, port_ann_vol, port_var_95_daily, port_sharpe = calculate_metrics(
        portfolio_assets, portfolio_returns, portfolio_cum, benchmark_returns, benchmark_ticker, risk_free_rate
    )

# -----------------------------
# Header KPIs
# -----------------------------
st.subheader("Portfolio Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Portfolio", "$50.0B MXN")
col2.metric("Assets", f"{len(portfolio_assets):,}")
col3.metric("Backtested Return", format_pct(port_ann_return))
col4.metric("Volatility", format_pct(port_ann_vol))
col5.metric("Sharpe", "—" if pd.isna(port_sharpe) else f"{port_sharpe:.2f}")

col6, col7, col8 = st.columns(3)
col6.metric("Daily VaR 95%", format_mxn(port_var_95_daily * TOTAL_PORTFOLIO_MXN if pd.notna(port_var_95_daily) else np.nan))
col7.metric("Benchmark", benchmark_ticker or "No data")
col8.metric("Vector sheet", f"{sheet_selected} / header row {header_row}")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_assets, tab_sovereign, tab_backtest, tab_tables, tab_export = st.tabs([
    "Overview", "Selected Assets", "Sovereign Bucket", "Backtesting & Risk", "Proxy Tables", "Export"
])

with tab_overview:
    pie_data = (
        portfolio_assets.groupby("Sleeve", as_index=False)
        .agg(Weight=("Weight", "sum"), Amount_MXN=("Amount MXN", "sum"), Assets=("ISIN", "count"))
        .sort_values("Weight", ascending=False)
    )
    pie_data["Weight %"] = pie_data["Weight"] * 100

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(pie_allocation_chart(pie_data), use_container_width=True)
    with c2:
        st.plotly_chart(bar_allocation_chart(pie_data), use_container_width=True)

    st.markdown("### IPS Allocation Check")
    st.dataframe(allocation_check.sort_values("Sleeve"), use_container_width=True)

with tab_assets:
    st.markdown("### Complete Selected Assets Table")
    sleeve_filter = st.multiselect("Filter by sleeve", sorted(portfolio_assets["Sleeve"].dropna().unique()), default=sorted(portfolio_assets["Sleeve"].dropna().unique()))
    filtered_assets = portfolio_assets[portfolio_assets["Sleeve"].isin(sleeve_filter)].copy()
    final_assets_table = filtered_assets[[
        "Sleeve", "Asset Name", "ISIN", "Tipo", "Emisora", "Serie",
        "Yield", "Yield Used", "Coupon", "Maturity", "Duration", "Price",
        "Rating", "Sector", "Weight %", "Amount MXN", "Yahoo Proxy", "Source",
    ]].copy()
    st.dataframe(final_assets_table, use_container_width=True, height=650)

with tab_sovereign:
    st.plotly_chart(sovereign_chart(sovereign_validation), use_container_width=True)
    st.markdown("### Sovereign Validation")
    st.dataframe(sovereign_validation, use_container_width=True)
    st.markdown("### Mandatory Government Instruments")
    gov_cols = [
        "ISIN", "Tipo", "Emisora", "Serie", "Maturity", "Duration", "Sovereign Bucket",
        "Yield Used", "Financial Score", "Weight within Sovereign Bucket", "Weight",
        "Coupon", "Price", "Rating", "Sector",
    ]
    st.dataframe(gov[[c for c in gov_cols if c in gov.columns]].sort_values("Weight", ascending=False), use_container_width=True)

with tab_backtest:
    if backtesting_table.empty:
        st.warning("Backtesting data could not be downloaded from Yahoo Finance. Check your internet connection or ticker availability.")
    else:
        st.plotly_chart(backtest_chart(backtesting_table, benchmark_ticker), use_container_width=True)
        st.plotly_chart(drawdown_chart(portfolio_cum), use_container_width=True)
        st.plotly_chart(returns_histogram(portfolio_returns, port_var_95_daily), use_container_width=True)
        fig_rr, sleeve_stats = sleeve_risk_return_chart(sleeve_returns, selected_proxy_table)
        st.plotly_chart(fig_rr, use_container_width=True)
        st.markdown("### Metrics")
        st.dataframe(metrics, use_container_width=True)
        st.markdown("### Backtesting Table")
        st.dataframe(backtesting_table.tail(250), use_container_width=True, height=500)

with tab_tables:
    st.markdown("### Proxy Policy Table")
    final_proxy_table = selected_proxy_table.merge(proxy_table, on="Sleeve", how="left")
    st.dataframe(final_proxy_table, use_container_width=True)

    st.markdown("### Other Proxy Assets Optimization")
    st.dataframe(other_assets, use_container_width=True)

    st.markdown("### Corporate Selection")
    corp_cols = ["ISIN", "Asset Name", "Tipo", "Emisora", "Serie", "Yield", "Coupon", "Maturity", "Duration", "Price", "Rating", "Sector", "Selection Score", "Weight"]
    st.dataframe(corp_selected[[c for c in corp_cols if c in corp_selected.columns]].sort_values("Weight", ascending=False), use_container_width=True)

with tab_export:
    st.markdown("### Download Results")
    final_assets_table_all = portfolio_assets[[
        "Sleeve", "Asset Name", "ISIN", "Tipo", "Emisora", "Serie",
        "Yield", "Yield Used", "Coupon", "Maturity", "Duration", "Price",
        "Rating", "Sector", "Weight %", "Amount MXN", "Yahoo Proxy", "Source",
    ]].copy()
    final_proxy_table = selected_proxy_table.merge(proxy_table, on="Sleeve", how="left")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        final_assets_table_all.to_excel(writer, sheet_name="Selected Assets", index=False)
        final_proxy_table.to_excel(writer, sheet_name="Proxy Table", index=False)
        allocation_check.to_excel(writer, sheet_name="Allocation Check", index=False)
        sovereign_validation.to_excel(writer, sheet_name="Sovereign Validation", index=False)
        if not backtesting_table.empty:
            backtesting_table.to_excel(writer, sheet_name="Backtesting", index=True)
        if not metrics.empty:
            metrics.to_excel(writer, sheet_name="Metrics", index=False)

    st.download_button(
        label="Download Excel output",
        data=output.getvalue(),
        file_name="IPS_Portfolio_75_79_Streamlit_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.caption("Note: Yahoo Finance proxies are used for backtesting where individual local bond tickers are not directly available.")
