# ============================================================
# Radar GARCH B3 - Web App Streamlit
# Estratégia: vender PUT no -1,5 desvio GARCH e comprar PUT no -2 desvios GARCH
# Uso no Streamlit Cloud: Main file path = app.py
# ============================================================

import math
import re
import warnings
from datetime import date, timedelta
from io import StringIO
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

APP_VERSION = "v5.1 - tabela Opções.net + decisão pronta"

# ============================================================
# Universo inicial
# ============================================================
UNIVERSE: Dict[str, Dict] = {
    "PETR4": {"setor": "Petróleo", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.95, "dividendos": "alto/cíclico"},
    "PETR3": {"setor": "Petróleo", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.90, "dividendos": "alto/cíclico"},
    "ITUB4": {"setor": "Banco", "banco": True, "bluechip": True, "carteira": True, "qualidade": 0.95, "dividendos": "recorrente"},
    "BBDC4": {"setor": "Banco", "banco": True, "bluechip": True, "carteira": True, "qualidade": 0.84, "dividendos": "recorrente"},
    "BBAS3": {"setor": "Banco", "banco": True, "bluechip": True, "carteira": True, "qualidade": 0.83, "dividendos": "alto/político"},
    "B3SA3": {"setor": "Bolsa", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.84, "dividendos": "moderado"},
    "ABEV3": {"setor": "Consumo defensivo", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.78, "dividendos": "moderado"},
    "GGBR4": {"setor": "Siderurgia", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.74, "dividendos": "cíclico"},
    "VALE3": {"setor": "Mineração", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.90, "dividendos": "alto/cíclico"},
    "CMIG4": {"setor": "Energia", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.78, "dividendos": "alto"},
    "CPLE6": {"setor": "Energia", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.76, "dividendos": "moderado/alto"},
    "TAEE11": {"setor": "Energia", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.82, "dividendos": "alto"},
    "BBSE3": {"setor": "Seguros", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.84, "dividendos": "alto"},
    "CXSE3": {"setor": "Seguros", "banco": False, "bluechip": False, "carteira": True, "qualidade": 0.72, "dividendos": "alto"},
    "VIVT3": {"setor": "Telecom", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.82, "dividendos": "moderado/alto"},
    "PSSA3": {"setor": "Seguros", "banco": False, "bluechip": True, "carteira": True, "qualidade": 0.82, "dividendos": "moderado"},
    "ITSA4": {"setor": "Holding financeira", "banco": True, "bluechip": True, "carteira": True, "qualidade": 0.80, "dividendos": "recorrente"},
    "SANB11": {"setor": "Banco", "banco": True, "bluechip": True, "carteira": True, "qualidade": 0.76, "dividendos": "recorrente"},
    "CSNA3": {"setor": "Siderurgia", "banco": False, "bluechip": False, "carteira": False, "qualidade": 0.55, "dividendos": "cíclico"},
    "USIM5": {"setor": "Siderurgia", "banco": False, "bluechip": False, "carteira": False, "qualidade": 0.50, "dividendos": "cíclico"},
}

# ============================================================
# Utilitários
# ============================================================
def brl(x) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"R$ {float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct_decimal(x) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.1f}%".replace(".", ",")


def num(x, nd=1) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.{nd}f}".replace(".", ",")


def clean_ticker(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(t).upper()).replace("SA", "")


def to_float(v) -> Optional[float]:
    if v is None or pd.isna(v):
        return None
    s = str(v).strip().replace("R$", "").replace("%", "").replace(" ", "")
    if s in {"", "-", "nan", "None"}:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def bounded(x, lo=0.0, hi=1.0):
    if x is None or pd.isna(x):
        return lo
    return max(lo, min(hi, float(x)))


def next_friday(ref: Optional[date] = None) -> date:
    ref = ref or date.today()
    days = (4 - ref.weekday()) % 7
    if days == 0:
        days = 7
    return ref + timedelta(days=days)


def business_days_until(expiry: date, ref: Optional[date] = None) -> int:
    ref = ref or date.today()
    start = ref if ref.weekday() < 5 else ref + timedelta(days=1)
    bdays = pd.bdate_range(start=start, end=expiry)
    return max(len(bdays), 1)


def round_down_strike(x: float, step: float) -> float:
    return math.floor(x / step) * step

# ============================================================
# Parser da tabela colada do Opções.net
# ============================================================
def parse_opcoesnet_assets_text(raw_text: str) -> pd.DataFrame:
    """Lê a tabela copiada da página /acoes do Opções.net.

    Espera linhas com este padrão aproximado:
    Ativo, Var. %, Últ., Data/Hora, CALL IV Rank, CALL Perc., CALL Vol. Impl.,
    Diff Vol. Calls/Puts, PUT IV Rank, PUT Perc., PUT Vol. Impl., HV Rank,
    HV Perc., Vol. Hist., Volume Financeiro.
    """
    if not raw_text:
        return pd.DataFrame()

    rows = []
    for line in raw_text.splitlines():
        line = line.strip("\ufeff\n\r ")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue

        ticker = parts[0].strip().upper()
        if not re.match(r"^[A-Z]{3,6}[0-9]{0,2}$|^[A-Z]{4}[0-9]{2}$|^IBOV$|^VXBR$", ticker):
            continue

        def get(i):
            return parts[i] if i < len(parts) else None

        rows.append({
            "ticker": clean_ticker(ticker),
            "opcoesnet_var_pct": to_float(get(1)),
            "opcoesnet_last": to_float(get(2)),
            "opcoesnet_date": get(3),
            "call_iv_rank": to_float(get(4)),
            "call_percentile": to_float(get(5)),
            "call_iv": to_float(get(6)),
            "diff_vol_calls_puts": to_float(get(7)),
            "put_iv_rank": to_float(get(8)),
            "put_percentile": to_float(get(9)),
            "put_iv": to_float(get(10)),
            "hv_rank": to_float(get(11)),
            "hv_percentile": to_float(get(12)),
            "hv": to_float(get(13)),
            "opcoesnet_fin_volume": to_float(get(14)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Volatilidades vêm como percentual. Transformar para decimal.
    for col in ["put_iv", "call_iv", "hv"]:
        df[col] = df[col].apply(lambda x: x / 100 if x is not None and not pd.isna(x) and x > 1.5 else x)
    return df


def get_asset_iv_row(asset_iv_df: pd.DataFrame, ticker: str) -> Dict:
    if asset_iv_df is None or asset_iv_df.empty:
        return {}
    df = asset_iv_df[asset_iv_df["ticker"].astype(str).apply(clean_ticker) == clean_ticker(ticker)]
    if df.empty:
        return {}
    return df.iloc[0].to_dict()

# ============================================================
# Tabela visual estilo Opções.net
# ============================================================
def var_cell_color(v):
    try:
        x = float(v)
    except Exception:
        return ""
    if pd.isna(x):
        return ""
    if x > 0:
        return "color: #0a8f2a; font-weight: 700;"
    if x < 0:
        return "color: #c62828; font-weight: 700;"
    return ""


def rank_cell_color(v):
    try:
        x = float(v)
    except Exception:
        return ""
    if pd.isna(x):
        return ""
    if x >= 90:
        return "background-color: #f7a8a8; color: #111;"
    if x >= 75:
        return "background-color: #f6c1c1; color: #111;"
    if x >= 60:
        return "background-color: #f9dede; color: #111;"
    if x <= 15:
        return "background-color: #c8c9ff; color: #111;"
    if x <= 30:
        return "background-color: #dedfff; color: #111;"
    if x <= 45:
        return "background-color: #eeeeff; color: #111;"
    return ""


def build_opcoesnet_view(asset_iv_df: pd.DataFrame, only_tickers=None) -> pd.DataFrame:
    if asset_iv_df is None or asset_iv_df.empty:
        return pd.DataFrame()
    df = asset_iv_df.copy()
    if only_tickers:
        allowed = {clean_ticker(t) for t in only_tickers}
        df = df[df["ticker"].astype(str).apply(clean_ticker).isin(allowed)]
    if df.empty:
        return pd.DataFrame()

    view = pd.DataFrame({
        "Ativo": df["ticker"],
        "Var. %": df["opcoesnet_var_pct"],
        "Últ.": df["opcoesnet_last"],
        "Data/Hora": df["opcoesnet_date"],
        "CALL IV Rank": df["call_iv_rank"],
        "CALL Perc.": df["call_percentile"],
        "CALL Vol. Impl.": df["call_iv"],
        "Diff Vol. Calls/Puts": df["diff_vol_calls_puts"],
        "PUT IV Rank": df["put_iv_rank"],
        "PUT Perc.": df["put_percentile"],
        "PUT Vol. Impl.": df["put_iv"],
        "HV Rank": df["hv_rank"],
        "HV Perc.": df["hv_percentile"],
        "Vol. Hist.": df["hv"],
        "Volume Financeiro": df["opcoesnet_fin_volume"],
    })
    return view.sort_values("PUT IV Rank", ascending=False, na_position="last").reset_index(drop=True)


def render_opcoesnet_assets_table(asset_iv_df: pd.DataFrame, only_tickers=None):
    view = build_opcoesnet_view(asset_iv_df, only_tickers)
    if view.empty:
        st.info("Cole a tabela 'Ativos com opções' do Opções.net para visualizar a tabela formatada.")
        return

    def fmt_dec(x, nd=1):
        if x is None or pd.isna(x):
            return ""
        return f"{float(x):.{nd}f}".replace(".", ",")

    def fmt_vol(x):
        if x is None or pd.isna(x):
            return ""
        return f"{float(x) * 100:.1f}".replace(".", ",")

    def fmt_brl_plain(x):
        if x is None or pd.isna(x):
            return ""
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def fmt_var(x):
        if x is None or pd.isna(x):
            return ""
        sign = "+" if float(x) > 0 else ""
        return f"{sign}{float(x):.2f}".replace(".", ",")

    color_cols = [
        "CALL IV Rank", "CALL Perc.", "CALL Vol. Impl.", "PUT IV Rank", "PUT Perc.",
        "PUT Vol. Impl.", "HV Rank", "HV Perc.", "Vol. Hist."
    ]
    styled = (
        view.style
        .format({
            "Var. %": fmt_var,
            "Últ.": fmt_brl_plain,
            "CALL IV Rank": fmt_dec,
            "CALL Perc.": fmt_dec,
            "CALL Vol. Impl.": fmt_vol,
            "Diff Vol. Calls/Puts": fmt_dec,
            "PUT IV Rank": fmt_dec,
            "PUT Perc.": fmt_dec,
            "PUT Vol. Impl.": fmt_vol,
            "HV Rank": fmt_dec,
            "HV Perc.": fmt_dec,
            "Vol. Hist.": fmt_vol,
            "Volume Financeiro": fmt_brl_plain,
        })
        .applymap(var_cell_color, subset=["Var. %"])
        .applymap(rank_cell_color, subset=[c for c in color_cols if c in view.columns])
        .set_properties(**{"text-align": "center", "font-size": "13px"})
        .set_properties(subset=["Ativo"], **{"font-weight": "700", "text-align": "left"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=520)
    st.download_button(
        "Baixar tabela tratada em CSV",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name="opcoesnet_ativos_tratado.csv",
        mime="text/csv",
    )

# ============================================================
# Dados, indicadores e GARCH
# ============================================================
@st.cache_data(ttl=60 * 60)
def download_history(ticker: str, period="3y") -> pd.DataFrame:
    yf_ticker = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
    df = yf.download(yf_ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(f"Sem dados para {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna(subset=["Close"])


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]

    out["EMA21"] = close.ewm(span=21, adjust=False).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["ATR_PCT"] = out["ATR14"] / close
    out["SUP20"] = low.rolling(20).min()
    out["VOL_FIN_21"] = (close * out["Volume"]).rolling(21).mean()
    return out


def garch_sigma_week(df: pd.DataFrame, horizon: int) -> Tuple[float, str]:
    close = df["Close"].dropna()
    returns = np.log(close / close.shift()).dropna()
    if len(returns) < 250:
        sigma_d = returns.ewm(span=60).std().iloc[-1]
        return float(sigma_d * math.sqrt(horizon)), "EWMA fallback"

    if ARCH_AVAILABLE:
        try:
            r_pct = returns * 100
            model = arch_model(r_pct, vol="GARCH", p=1, q=1, mean="Constant", dist="normal")
            res = model.fit(disp="off")
            fc = res.forecast(horizon=horizon, reindex=False)
            var_pct = fc.variance.values[-1]
            sigma = math.sqrt(float(np.sum(var_pct))) / 100.0
            return sigma, "GARCH(1,1)"
        except Exception:
            pass

    sigma_d = returns.ewm(span=60).std().iloc[-1]
    return float(sigma_d * math.sqrt(horizon)), "EWMA fallback"


def choose_strikes(price: float, sigma: float, step: float) -> Dict[str, float]:
    b15 = price * math.exp(-1.5 * sigma)
    b20 = price * math.exp(-2.0 * sigma)
    up15 = price * math.exp(1.5 * sigma)
    up20 = price * math.exp(2.0 * sigma)
    sell_put = round_down_strike(b15, step)
    buy_put = round_down_strike(b20, step)
    if buy_put >= sell_put:
        buy_put = max(sell_put - step, step)
    return {
        "banda_menos_15": b15,
        "banda_menos_20": b20,
        "banda_mais_15": up15,
        "banda_mais_20": up20,
        "sell_put": sell_put,
        "buy_put": buy_put,
        "width": max(sell_put - buy_put, 0.0),
    }

# ============================================================
# Black-Scholes para estimar prêmio quando só há IV das PUTs
# ============================================================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

# ============================================================
# Scoring, decisão e análise
# ============================================================
def trend_score(last: pd.Series) -> float:
    score = 0.0
    price = float(last["Close"])
    if price > float(last.get("EMA21", price)):
        score += 0.25
    if price > float(last.get("EMA50", price)):
        score += 0.25
    if price > float(last.get("EMA200", price)):
        score += 0.25
    rsi = float(last.get("RSI14", 50))
    if rsi >= 40:
        score += 0.15
    if 45 <= rsi <= 70:
        score += 0.10
    return bounded(score)


def iv_score(iv_rank: float, min_ivr: float, max_ivr: float) -> Tuple[float, str]:
    if iv_rank is None or pd.isna(iv_rank):
        return 0.50, "sem IV Rank; conferir manualmente"
    if iv_rank < min_ivr:
        return 0.15, "IV Rank baixo: prêmio pode ser fraco"
    if iv_rank > max_ivr:
        return 0.45, "IV Rank alto: verificar evento/balanço/notícia"
    mid = (min_ivr + max_ivr) / 2
    spread = max((max_ivr - min_ivr) / 2, 1)
    return bounded(1 - abs(iv_rank - mid) / spread * 0.35), "IV Rank dentro da faixa"


def decision_from_metrics(row: Dict, settings: Dict) -> Tuple[str, str]:
    reasons = []
    if row["price"] > settings["max_price"]:
        return "EVITAR", "Preço acima do limite definido."
    if row["avg_fin_vol"] < settings["min_fin_vol"]:
        return "EVITAR", "Liquidez do ativo abaixo do mínimo."
    if row["trend_score"] < 0.35:
        return "EVITAR", "Tendência fraca: evitar vender put em ativo caindo."
    if row["atr_pct"] > settings["max_atr_pct"]:
        reasons.append("ATR alto; reduzir quantidade ou aguardar")
    if row["credit_width"] is None or pd.isna(row["credit_width"]):
        return "AGUARDAR", "Sem estimativa de prêmio. Cole a tabela do Opções.net ou confira manualmente."
    if row["credit_width"] < settings["min_credit_width"]:
        return "AGUARDAR", "Crédito da trava não compensa a largura/risco."
    if row.get("credit_is_estimated", False):
        reasons.append("crédito estimado pela IV; confirmar bid/ask no home broker")
    if row["iv_rank"] is not None and not pd.isna(row["iv_rank"]):
        if row["iv_rank"] < settings["min_ivr"]:
            return "AGUARDAR", "IV Rank baixo: prêmio tende a ser fraco."
        if row["iv_rank"] > settings["max_ivr"]:
            reasons.append("IV Rank alto: verificar evento/balanço/notícia")
    if row["score"] >= 75:
        return "OPERAR SEGUNDA", "; ".join(reasons) if reasons else "Passou nos filtros principais. Conferir ordem no home broker."
    if row["portfolio_ok"]:
        return "CARTEIRA SE EXERCIDO", "Ativo bom para carteira, mas operação não está perfeita para renda nesta semana."
    return "AGUARDAR", "; ".join(reasons) if reasons else "Ativo razoável, mas score ainda não justifica decisão automática."


def analyze_ticker(ticker: str, meta: Dict, settings: Dict, asset_iv_df: pd.DataFrame) -> Dict:
    df = add_indicators(download_history(ticker, period=settings["history_period"]))
    last = df.iloc[-1]
    price = float(last["Close"])
    horizon = business_days_until(settings["expiry_date"])
    sigma, model_name = garch_sigma_week(df, horizon)
    strikes = choose_strikes(price, sigma, settings["strike_step"])

    sell_put = strikes["sell_put"]
    buy_put = strikes["buy_put"]
    width = strikes["width"]

    asset_iv = get_asset_iv_row(asset_iv_df, ticker)
    iv_rank = asset_iv.get("put_iv_rank", np.nan) if asset_iv else np.nan
    put_percentile = asset_iv.get("put_percentile", np.nan) if asset_iv else np.nan
    put_iv = asset_iv.get("put_iv", np.nan) if asset_iv else np.nan
    hv = asset_iv.get("hv", np.nan) if asset_iv else np.nan
    hv_rank = asset_iv.get("hv_rank", np.nan) if asset_iv else np.nan

    # Estimativa teórica do crédito pela IV das PUTs do Opções.net.
    # Não substitui bid/ask real da corretora.
    credit = np.nan
    sell_price = np.nan
    buy_price = np.nan
    credit_is_estimated = False
    if put_iv is not None and not pd.isna(put_iv) and put_iv > 0 and width > 0:
        T = max(settings["horizon_days_for_pricing"] / 252.0, 1 / 252)
        r = settings["risk_free_rate"]
        sell_price = bs_put_price(price, sell_put, T, r, float(put_iv))
        buy_price = bs_put_price(price, buy_put, T, r, float(put_iv))
        credit = max(sell_price - buy_price, np.nan)
        credit_is_estimated = bool(credit is not None and not pd.isna(credit) and credit > 0)

    credit_width = (credit / width) if width and credit is not None and not pd.isna(credit) else np.nan
    estimated_credit_for_risk = settings["min_credit_width"] * width
    credit_for_risk = float(credit) if credit is not None and not pd.isna(credit) and credit > 0 else estimated_credit_for_risk
    max_loss_ps = max(width - credit_for_risk, 0.0)

    avg_fin_vol = float(last.get("VOL_FIN_21", np.nan))
    atr_pct = float(last.get("ATR_PCT", np.nan))
    sup20 = float(last.get("SUP20", np.nan))
    dist_sell = (price - sell_put) / price if price > 0 else np.nan

    tscore = trend_score(last)
    liquidity_score = bounded(avg_fin_vol / settings["full_liquidity"] if not pd.isna(avg_fin_vol) else 0)
    distance_score = bounded(dist_sell / settings["target_distance"] if not pd.isna(dist_sell) else 0)
    atr_score = bounded(1 - (atr_pct / settings["max_atr_pct"])) if not pd.isna(atr_pct) else 0.5
    support_bonus = 1.0 if not pd.isna(sup20) and sell_put <= sup20 else 0.5
    ivs, iv_msg = iv_score(iv_rank, settings["min_ivr"], settings["max_ivr"])
    premium_score = 0.5 if pd.isna(credit_width) else bounded(credit_width / 0.30)
    quality_score = float(meta.get("qualidade", 0.5))

    score = (
        18 * liquidity_score +
        18 * tscore +
        14 * distance_score +
        10 * atr_score +
        10 * support_bonus +
        14 * ivs +
        12 * premium_score +
        14 * quality_score
    )

    qty_by_loss = math.floor(settings["risk_per_asset"] / (max_loss_ps * 100)) if max_loss_ps > 0 else 0
    qty_by_exercise = math.floor(settings["max_exercise_per_asset"] / (sell_put * 100)) if sell_put > 0 else 0
    qty = max(0, min(qty_by_loss, qty_by_exercise, settings["max_contracts_per_asset"]))

    max_gain = (float(credit) if not pd.isna(credit) else estimated_credit_for_risk) * 100 * qty
    max_loss = max_loss_ps * 100 * qty
    effective_if_exercised = sell_put - (float(credit) if not pd.isna(credit) else 0.0)
    portfolio_ok = bool(meta.get("carteira")) and float(meta.get("qualidade", 0)) >= settings["min_quality_portfolio"]

    row = {
        "ticker": ticker,
        "setor": meta.get("setor", ""),
        "banco": bool(meta.get("banco", False)),
        "dividendos": meta.get("dividendos", ""),
        "qualidade": float(meta.get("qualidade", 0)),
        "portfolio_ok": portfolio_ok,
        "price": price,
        "avg_fin_vol": avg_fin_vol,
        "horizon": horizon,
        "sigma_week": sigma,
        "model": model_name,
        "band_minus_15": strikes["banda_menos_15"],
        "band_minus_20": strikes["banda_menos_20"],
        "band_plus_15": strikes["banda_mais_15"],
        "band_plus_20": strikes["banda_mais_20"],
        "sell_put": sell_put,
        "buy_put": buy_put,
        "width": width,
        "sell_price": sell_price,
        "buy_price": buy_price,
        "credit": credit,
        "credit_width": credit_width,
        "credit_is_estimated": credit_is_estimated,
        "max_loss_ps": max_loss_ps,
        "iv": put_iv,
        "iv_rank": iv_rank,
        "put_percentile": put_percentile,
        "hv": hv,
        "hv_rank": hv_rank,
        "dist_sell": dist_sell,
        "atr_pct": atr_pct,
        "rsi": float(last.get("RSI14", np.nan)),
        "ema21": float(last.get("EMA21", np.nan)),
        "ema50": float(last.get("EMA50", np.nan)),
        "ema200": float(last.get("EMA200", np.nan)),
        "support20": sup20,
        "trend_score": tscore,
        "score": score,
        "qty": qty,
        "max_gain": max_gain,
        "max_loss": max_loss,
        "effective_if_exercised": effective_if_exercised,
        "iv_msg": iv_msg,
        "option_source": "Tabela Ativos Opções.net" if asset_iv else "sem tabela Opções.net",
    }

    dec, reason = decision_from_metrics(row, settings)
    row["decision"] = dec
    row["reason"] = reason
    row["order_text"] = (
        f"{ticker}: vender PUT {sell_put:.2f} e comprar PUT {buy_put:.2f} "
        f"venc. {settings['expiry_date'].strftime('%d/%m/%Y')} | qtd sugerida: {qty} trava(s)"
    )
    row["roll_plan"] = (
        "Plano: se ameaçar o strike vendido, rolar para baixo e para frente até 3 vezes. "
        "Na 4ª ameaça, se ainda for ativo de carteira, vender/zerar a put comprada e aceitar possível exercício; "
        "se a tese piorar, encerrar a trava."
    )
    return row

# ============================================================
# Interface Streamlit
# ============================================================
st.set_page_config(page_title="Radar GARCH B3", page_icon="📈", layout="wide")
st.title("📈 Radar GARCH B3 — Travas de Put com Decisão Pronta")
st.caption(f"{APP_VERSION} | Venda da put no -1,5σ GARCH e compra da put no -2σ GARCH")

with st.sidebar:
    st.header("Configuração")
    margin = st.number_input("Margem/capital de referência", min_value=1000.0, value=100000.0, step=1000.0)
    max_price = st.number_input("Preço máximo do ativo", min_value=1.0, value=50.0, step=1.0)
    max_banks = st.number_input("Máximo de bancos na lista final", min_value=0, value=2, step=1)
    risk_per_asset_pct = st.slider("Risco máximo por ativo", 0.5, 10.0, 3.0, 0.5) / 100
    exercise_per_asset_pct = st.slider("Máximo que aceito comprar por ativo se exercido", 5.0, 40.0, 20.0, 1.0) / 100
    max_contracts_per_asset = st.number_input("Máximo de travas por ativo", min_value=1, value=20, step=1)

    st.divider()
    expiry = st.date_input("Vencimento semanal alvo", value=next_friday())
    strike_step = st.selectbox("Intervalo de strike", [0.10, 0.25, 0.50, 1.00], index=2)
    min_credit_width = st.slider("Crédito mínimo / largura da trava", 0.05, 0.40, 0.20, 0.01)
    min_ivr = st.slider("IV Rank mínimo", 0, 100, 35, 5)
    max_ivr = st.slider("IV Rank máximo", 0, 100, 80, 5)

    st.divider()
    min_fin_vol = st.number_input("Liquidez mínima à vista/dia", min_value=0.0, value=50_000_000.0, step=10_000_000.0)
    full_liquidity = st.number_input("Liquidez para score cheio", min_value=1_000_000.0, value=500_000_000.0, step=50_000_000.0)
    max_atr_pct = st.slider("ATR% máximo aceitável", 0.01, 0.12, 0.06, 0.005)
    target_distance = st.slider("Distância ideal da put vendida", 0.02, 0.20, 0.08, 0.01)
    min_quality_portfolio = st.slider("Qualidade mínima para virar carteira", 0.40, 0.95, 0.75, 0.05)
    history_period = st.selectbox("Histórico usado", ["1y", "2y", "3y", "5y"], index=2)

    st.divider()
    selected_tickers = st.multiselect("Ativos analisados", list(UNIVERSE.keys()), default=list(UNIVERSE.keys()))
    pasted_options_text = st.text_area(
        "Cole aqui a tabela 'Ativos com opções' do Opções.net",
        height=160,
        placeholder="Cole a tabela copiada da página opcoes.net.br/acoes..."
    )
    risk_free_rate = st.number_input("Taxa livre de risco anual p/ estimar prêmio", min_value=0.0, max_value=0.30, value=0.105, step=0.005)

settings = {
    "margin": margin,
    "max_price": max_price,
    "max_banks": int(max_banks),
    "risk_per_asset": margin * risk_per_asset_pct,
    "max_exercise_per_asset": margin * exercise_per_asset_pct,
    "max_contracts_per_asset": int(max_contracts_per_asset),
    "expiry_date": expiry,
    "strike_step": float(strike_step),
    "min_credit_width": float(min_credit_width),
    "min_ivr": float(min_ivr),
    "max_ivr": float(max_ivr),
    "min_fin_vol": float(min_fin_vol),
    "full_liquidity": float(full_liquidity),
    "max_atr_pct": float(max_atr_pct),
    "target_distance": float(target_distance),
    "min_quality_portfolio": float(min_quality_portfolio),
    "history_period": history_period,
    "risk_free_rate": float(risk_free_rate),
    "horizon_days_for_pricing": business_days_until(expiry),
}

st.info(
    "Use na sexta após o fechamento para gerar o radar da próxima semana. "
    "Na segunda após 10h30, rode novamente e confirme prêmios, vencimento, spread e liquidez no home broker."
)

with st.expander("Como copiar a tabela do Opções.net"):
    st.markdown(
        """
1. Entre logado no **Opções.net**.  
2. Abra a tela **Ações / Ativos com opções**.  
3. Selecione a tabela inteira e copie.  
4. Cole no campo da barra lateral.  
5. Clique em **Gerar decisão pronta**.

A tabela colada permite ao app usar **IV Rank das PUTs, Percentil, Volatilidade Implícita e Volume Financeiro**.
        """
    )

if st.button("🚀 Gerar decisão pronta", type="primary"):
    asset_iv_df = parse_opcoesnet_assets_text(pasted_options_text)
    tickers = [t for t in selected_tickers if t in UNIVERSE]

    if not asset_iv_df.empty:
        st.success(f"Tabela do Opções.net carregada: {len(asset_iv_df)} ativos com IV/IV Rank de PUTs.")
        with st.expander("📋 Ver tabela igual ao Opções.net dentro do app", expanded=True):
            escopo = st.radio(
                "Mostrar tabela",
                ["Apenas ativos analisados", "Tabela completa colada"],
                horizontal=True,
                key="opcoesnet_table_scope",
            )
            render_opcoesnet_assets_table(
                asset_iv_df,
                only_tickers=tickers if escopo == "Apenas ativos analisados" else None,
            )
    else:
        st.warning("Tabela do Opções.net não foi carregada. O app calculará GARCH, mas ficará sem IV Rank/IV para decisão completa.")

    rows = []
    errors = []
    progress = st.progress(0)

    for i, ticker in enumerate(tickers):
        try:
            row = analyze_ticker(ticker, UNIVERSE[ticker], settings, asset_iv_df)
            rows.append(row)
        except Exception as e:
            errors.append({"ticker": ticker, "erro": str(e)})
        progress.progress((i + 1) / max(len(tickers), 1))

    if not rows:
        st.error("Nenhum ativo analisado com sucesso.")
        if errors:
            st.json(errors)
        st.stop()

    df = pd.DataFrame(rows)
    df = df[df["price"] <= max_price].copy()

    # Limita a quantidade de bancos na lista final.
    banks = df[df["banco"]].sort_values("score", ascending=False).head(int(max_banks))
    non_banks = df[~df["banco"]]
    final = pd.concat([banks, non_banks], ignore_index=True).sort_values("score", ascending=False)

    st.subheader("✅ Decisão pronta")
    display = final.copy()
    display["Score"] = display["score"].round(1)
    display["Preço"] = display["price"].map(brl)
    display["Vender Put"] = display["sell_put"].map(brl)
    display["Comprar Put"] = display["buy_put"].map(brl)
    display["Banda -1,5σ"] = display["band_minus_15"].map(brl)
    display["Banda -2σ"] = display["band_minus_20"].map(brl)
    display["Crédito estimado"] = display["credit"].map(brl)
    display["Crédito/Largura"] = display["credit_width"].map(lambda x: pct_decimal(x) if not pd.isna(x) else "sem dado")
    display["IV Rank"] = display["iv_rank"].map(lambda x: num(x, 0) if not pd.isna(x) else "sem dado")
    display["Perc. PUT"] = display["put_percentile"].map(lambda x: num(x, 0) if not pd.isna(x) else "sem dado")
    display["Vol. Impl. PUT"] = display["iv"].map(lambda x: pct_decimal(x) if not pd.isna(x) else "sem dado")
    display["ATR%"] = display["atr_pct"].map(pct_decimal)
    display["Distância Put"] = display["dist_sell"].map(pct_decimal)
    display["Qtd"] = display["qty"]
    display["Ganho Máx"] = display["max_gain"].map(brl)
    display["Perda Máx"] = display["max_loss"].map(brl)
    display["Preço efetivo se exercido"] = display["effective_if_exercised"].map(brl)

    cols = [
        "decision", "ticker", "setor", "Score", "Preço", "Vender Put", "Comprar Put",
        "Crédito estimado", "Crédito/Largura", "IV Rank", "Perc. PUT", "Vol. Impl. PUT",
        "Qtd", "Ganho Máx", "Perda Máx", "Preço efetivo se exercido", "reason",
    ]
    st.dataframe(
        display[cols].rename(columns={
            "decision": "Decisão",
            "ticker": "Ativo",
            "setor": "Setor",
            "reason": "Motivo",
        }),
        use_container_width=True,
        hide_index=True,
    )

    top_operar = final[final["decision"].eq("OPERAR SEGUNDA")]
    if not top_operar.empty:
        st.success("Há candidatos para olhar na segunda. Confirme vencimento, bid/ask, liquidez e prêmio real no home broker.")
    else:
        st.warning("Nenhum ativo recebeu decisão automática de OPERAR. O app está sendo conservador.")

    st.subheader("📌 Detalhe do ativo")
    choice = st.selectbox("Escolha um ativo para ver a ordem e o plano", final["ticker"].tolist())
    row = final[final["ticker"] == choice].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decisão", row["decision"])
    c2.metric("Score", f"{row['score']:.1f}")
    c3.metric("Quantidade", int(row["qty"]))
    c4.metric("Preço efetivo", brl(row["effective_if_exercised"]))

    st.markdown(f"### Ordem sugerida\n**{row['order_text']}**")
    st.write(f"**Motivo:** {row['reason']}")
    st.write(f"**Plano de proteção/rolagem:** {row['roll_plan']}")
    st.write(f"**Fonte dos dados de opção:** {row['option_source']}. O crédito é estimado pela IV das PUTs; confirme prêmio real no home broker.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Banda -1,5σ", brl(row["band_minus_15"]))
    k2.metric("Banda -2σ", brl(row["band_minus_20"]))
    k3.metric("GARCH semanal", pct_decimal(row["sigma_week"]))
    k4.metric("Modelo", row["model"])

    if PLOTLY_AVAILABLE:
        try:
            chart = add_indicators(download_history(choice, settings["history_period"])).tail(140)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=chart.index,
                open=chart["Open"],
                high=chart["High"],
                low=chart["Low"],
                close=chart["Close"],
                name=choice,
            ))
            fig.add_hline(y=row["sell_put"], line_dash="dash", annotation_text="Put vendida")
            fig.add_hline(y=row["buy_put"], line_dash="dot", annotation_text="Put comprada")
            fig.add_hline(y=row["band_minus_15"], line_dash="dash", annotation_text="Banda -1,5σ")
            fig.add_hline(y=row["band_minus_20"], line_dash="dot", annotation_text="Banda -2σ")
            fig.update_layout(height=520, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    st.subheader("🧮 Simulador rápido de prêmio real")
    s1, s2, s3 = st.columns(3)
    psell = s1.number_input(
        "Prêmio real da put vendida",
        min_value=0.0,
        value=float(row["sell_price"]) if not pd.isna(row["sell_price"]) else 0.20,
        step=0.01,
    )
    pbuy = s2.number_input(
        "Prêmio real da put comprada",
        min_value=0.0,
        value=float(row["buy_price"]) if not pd.isna(row["buy_price"]) else 0.05,
        step=0.01,
    )
    qty_manual = s3.number_input("Quantidade", min_value=1, value=max(int(row["qty"]), 1), step=1)

    credit_real = psell - pbuy
    width = float(row["width"])
    max_loss_ps = max(width - credit_real, 0)
    st.write(
        f"Crédito líquido: **{brl(credit_real)}** | "
        f"Crédito/Largura: **{pct_decimal(credit_real / width if width else np.nan)}** | "
        f"Perda máxima: **{brl(max_loss_ps * 100 * qty_manual)}** | "
        f"Ganho máximo: **{brl(credit_real * 100 * qty_manual)}**"
    )

    if credit_real <= 0:
        st.error("Não montar: crédito líquido zero ou negativo.")
    elif credit_real / width < settings["min_credit_width"]:
        st.warning("Aguardar: prêmio real não compensa a largura da trava.")
    else:
        st.success("Prêmio real compensa pela regra configurada. Conferir liquidez/spread antes da ordem.")

    with st.expander("Todos os dados calculados"):
        st.dataframe(final, use_container_width=True, hide_index=True)

    if errors:
        with st.expander("Erros/ativos ignorados"):
            st.json(errors)
else:
    st.info("Clique em **Gerar decisão pronta** para calcular o radar.")

st.divider()
st.caption(
    "Aviso: ferramenta educacional e quantitativa. Não envia ordens e não garante lucro. "
    "Quando o prêmio for estimado pela IV do Opções.net, confirme prêmios reais, vencimentos, bid/ask, liquidez, eventos corporativos e regras da corretora antes de operar."
)
