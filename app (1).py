# ============================================================
# Radar GARCH B3 - Travas de Put com Decisão Pronta
# Estratégia: vender put no -1,5 desvio GARCH e comprar put no -2 desvios GARCH
# Uso: streamlit run app.py
# ============================================================

import math
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

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

APP_VERSION = "v3.0 - decisão pronta"
OPCOES_BASE = "https://opcoes.net.br"

# Universo inicial: ativos líquidos/blue chips/dividendos e alguns candidatos abaixo de R$50.
# O app filtra preço, liquidez, tendência, IV, prêmio e limite de bancos.
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
    # Cíclicas/voláteis: entram, mas com qualidade menor. O app pode evitar se filtros ruins.
    "CSNA3": {"setor": "Siderurgia", "banco": False, "bluechip": False, "carteira": False, "qualidade": 0.55, "dividendos": "cíclico"},
    "USIM5": {"setor": "Siderurgia", "banco": False, "bluechip": False, "carteira": False, "qualidade": 0.50, "dividendos": "cíclico"},
}

# -----------------------------
# Formatação
# -----------------------------
def brl(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{100*x:.1f}%".replace(".", ",")


def num(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{nd}f}".replace(".", ",")


def clean_ticker(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(t).upper()).replace("SA", "")


def normalize_col(c: str) -> str:
    s = str(c).strip().lower()
    s = (s.replace("ç", "c").replace("ã", "a").replace("á", "a").replace("à", "a")
         .replace("â", "a").replace("é", "e").replace("ê", "e").replace("í", "i")
         .replace("ó", "o").replace("ô", "o").replace("ú", "u"))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def to_float(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().replace("R$", "").replace("%", "").replace(" ", "")
    if s in {"", "-", "nan", "None"}:
        return None
    # formato BR: 1.234,56
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

# -----------------------------
# Datas
# -----------------------------
def next_friday(ref: Optional[date] = None) -> date:
    ref = ref or date.today()
    days = (4 - ref.weekday()) % 7
    if days == 0:
        # Se for sexta, o radar de sexta pós-fechamento deve mirar a próxima sexta.
        days = 7
    return ref + timedelta(days=days)


def business_days_until(expiry: date, ref: Optional[date] = None) -> int:
    ref = ref or date.today()
    start = ref + timedelta(days=1) if ref.weekday() >= 5 else ref
    bdays = pd.bdate_range(start=start, end=expiry)
    return max(len(bdays), 1)

# -----------------------------
# Dados e indicadores
# -----------------------------
@st.cache_data(ttl=60 * 60)
def download_history(ticker: str, period="3y") -> pd.DataFrame:
    yf_ticker = ticker if ticker.endswith(".SA") else f"{ticker}.SA"
    df = yf.download(yf_ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(f"Sem dados para {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna(subset=["Close"])
    return df


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
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    out["ATR_PCT"] = out["ATR14"] / close
    out["SUP20"] = low.rolling(20).min()
    out["VOL_FIN_21"] = (close * out["Volume"]).rolling(21).mean()
    return out


def garch_sigma_week(df: pd.DataFrame, horizon: int) -> Tuple[float, str]:
    close = df["Close"].dropna()
    returns = np.log(close / close.shift()).dropna()
    if len(returns) < 300:
        # fallback com EWMA se histórico curto
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


def round_down_strike(x: float, step: float) -> float:
    return math.floor(x / step) * step


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

# -----------------------------
# Dados públicos Opções.net / CSV
# -----------------------------
@st.cache_data(ttl=60 * 30)
def fetch_opcoesnet_tables(ticker: str) -> pd.DataFrame:
    """Best-effort: tenta ler tabelas públicas do Opções.net.
    Observação: o site pode carregar parte por JavaScript e mudar layout. Por isso o app também aceita CSV.
    """
    urls = [
        f"{OPCOES_BASE}/opcoes/bovespa/{ticker}",
        f"{OPCOES_BASE}/opcoes/bovespa?ativo={ticker}",
        f"{OPCOES_BASE}/{ticker}",
    ]
    frames = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=12)
            if r.status_code != 200 or not r.text:
                continue
            tables = pd.read_html(StringIO(r.text))
            for tb in tables:
                if tb is not None and not tb.empty:
                    tb["_fonte_url"] = url
                    frames.append(tb)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.columns = [normalize_col(c) for c in df.columns]
    df["ticker"] = ticker
    return df


def parse_uploaded_options(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    try:
        raw = file.read()
        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(StringIO(raw.decode("utf-8", errors="ignore")), sep=sep)
                if df.shape[1] > 1:
                    break
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            return df
        df.columns = [normalize_col(c) for c in df.columns]
        if "ticker" not in df.columns and "ativo" in df.columns:
            df["ticker"] = df["ativo"]
        df["ticker"] = df.get("ticker", "").apply(clean_ticker)
        return df
    except Exception:
        return pd.DataFrame()


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df.empty:
        return None
    cols = set(df.columns)
    for c in candidates:
        c2 = normalize_col(c)
        if c2 in cols:
            return c2
    # busca parcial
    for col in df.columns:
        for cand in candidates:
            if normalize_col(cand) in col:
                return col
    return None


def match_option_pair(options_df: pd.DataFrame, ticker: str, sell_strike: float, buy_strike: float) -> Dict:
    """Seleciona preços/IV/volume da put vendida/comprada a partir de CSV ou tabela pública normalizada."""
    result = {
        "sell_price": np.nan, "buy_price": np.nan, "credit": np.nan,
        "iv": np.nan, "iv_rank": np.nan, "volume_option": np.nan, "source": "sem dados"
    }
    if options_df is None or options_df.empty:
        return result
    df = options_df.copy()
    if "ticker" in df.columns:
        df = df[df["ticker"].astype(str).str.upper().str.replace(".SA", "", regex=False).apply(clean_ticker) == ticker]
    if df.empty:
        return result

    strike_col = find_col(df, ["strike", "preco_exercicio", "exercicio", "preco_de_exercicio"])
    type_col = find_col(df, ["tipo", "call_put", "cp", "opcao_tipo"])
    last_col = find_col(df, ["ultimo", "ultima", "last", "preco", "cotacao", "fechamento"])
    bid_col = find_col(df, ["bid", "compra"])
    ask_col = find_col(df, ["ask", "venda"])
    iv_col = find_col(df, ["iv", "volatilidade_implicita", "vi"])
    ivr_col = find_col(df, ["iv_rank", "rank", "ivrank"])
    vol_col = find_col(df, ["volume", "volume_financeiro", "negocios", "qtd_negocios"])

    if strike_col is None:
        return result

    df["_strike"] = df[strike_col].apply(to_float)
    df = df.dropna(subset=["_strike"])
    if type_col:
        # mantém PUT quando a informação existir
        mask_put = df[type_col].astype(str).str.upper().str.contains("P|PUT|VENDA", regex=True, na=False)
        if mask_put.any():
            df = df[mask_put]
    if df.empty:
        return result

    def nearest_row(strike):
        idx = (df["_strike"] - strike).abs().idxmin()
        return df.loc[idx]

    sell = nearest_row(sell_strike)
    buy = nearest_row(buy_strike)

    def px(row):
        bid = to_float(row.get(bid_col)) if bid_col else None
        ask = to_float(row.get(ask_col)) if ask_col else None
        last = to_float(row.get(last_col)) if last_col else None
        # Para venda, o realista seria bid; para compra, ask. Se não tiver, usa último.
        return bid, ask, last

    s_bid, s_ask, s_last = px(sell)
    b_bid, b_ask, b_last = px(buy)
    sell_price = s_bid if s_bid is not None and s_bid > 0 else s_last
    buy_price = b_ask if b_ask is not None and b_ask > 0 else b_last

    if sell_price is not None:
        result["sell_price"] = float(sell_price)
    if buy_price is not None:
        result["buy_price"] = float(buy_price)
    if sell_price is not None and buy_price is not None:
        result["credit"] = float(sell_price - buy_price)

    ivs = []
    if iv_col:
        for row in [sell, buy]:
            val = to_float(row.get(iv_col))
            if val is not None:
                if val > 1.5:  # se veio em %, transforma para decimal
                    val = val / 100
                ivs.append(val)
    if ivs:
        result["iv"] = float(np.nanmean(ivs))

    if ivr_col:
        val = to_float(sell.get(ivr_col))
        if val is not None:
            result["iv_rank"] = float(val)

    if vol_col:
        vals = []
        for row in [sell, buy]:
            val = to_float(row.get(vol_col))
            if val is not None:
                vals.append(val)
        if vals:
            result["volume_option"] = float(np.nanmean(vals))

    result["source"] = "CSV/Opções.net público"
    return result

# -----------------------------
# Scoring e decisão
# -----------------------------
def bounded(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def trend_score(last: pd.Series) -> float:
    score = 0.0
    price = float(last["Close"])
    if price > float(last.get("EMA21", price)): score += 0.25
    if price > float(last.get("EMA50", price)): score += 0.25
    if price > float(last.get("EMA200", price)): score += 0.25
    rsi = float(last.get("RSI14", 50))
    if rsi >= 40: score += 0.15
    if 45 <= rsi <= 70: score += 0.10
    return bounded(score)


def iv_score(iv_rank: float, min_ivr: float, max_ivr: float) -> Tuple[float, str]:
    if np.isnan(iv_rank):
        return 0.50, "sem IV Rank público; conferir manualmente"
    if iv_rank < min_ivr:
        return 0.15, "IV Rank baixo: prêmio pode não compensar"
    if iv_rank > max_ivr:
        return 0.45, "IV Rank alto: prêmio bom, mas pode ter risco/evento"
    # melhor no miolo da faixa
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
    if row["credit_width"] is not None and not np.isnan(row["credit_width"]):
        if row["credit_width"] < settings["min_credit_width"]:
            return "AGUARDAR", "Crédito da trava não compensa o risco."
    else:
        return "AGUARDAR", "Sem prêmio real da opção. Conferir Opções.net/home broker ou importar CSV."
    if row["iv_rank"] is not None and not np.isnan(row["iv_rank"]):
        if row["iv_rank"] < settings["min_ivr"]:
            return "AGUARDAR", "IV Rank baixo: prêmio tende a ser fraco."
        if row["iv_rank"] > settings["max_ivr"]:
            reasons.append("IV Rank alto: verificar evento/balanço/notícia")
    if row["score"] >= 75:
        return "OPERAR SEGUNDA", "; ".join(reasons) if reasons else "Passou nos filtros principais. Conferir ordem no home broker."
    if row["portfolio_ok"]:
        return "CARTEIRA SE EXERCIDO", "Ativo serve para carteira, mas operação não está ideal para renda nesta semana."
    return "AGUARDAR", "; ".join(reasons) if reasons else "Ativo razoável, mas score ainda não justifica decisão automática."


def analyze_ticker(ticker: str, meta: Dict, settings: Dict, options_df: pd.DataFrame) -> Dict:
    df = add_indicators(download_history(ticker, period=settings["history_period"]))
    last = df.iloc[-1]
    price = float(last["Close"])
    horizon = business_days_until(settings["expiry_date"])
    sigma, model_name = garch_sigma_week(df, horizon)
    strikes = choose_strikes(price, sigma, settings["strike_step"])
    sell_put = strikes["sell_put"]
    buy_put = strikes["buy_put"]
    width = strikes["width"]

    opt = match_option_pair(options_df, ticker, sell_put, buy_put)
    credit = opt.get("credit", np.nan)
    # Se não tiver preço real, usa crédito estimado apenas para simular quantidade, mas não aprova operar.
    estimated_credit = settings["min_credit_width"] * width
    credit_for_risk = float(credit) if credit is not None and not np.isnan(credit) and credit > 0 else estimated_credit
    max_loss_ps = max(width - credit_for_risk, 0.0)
    credit_width = (float(credit) / width) if width and credit is not None and not np.isnan(credit) else np.nan

    avg_fin_vol = float(last.get("VOL_FIN_21", np.nan))
    atr_pct = float(last.get("ATR_PCT", np.nan))
    sup20 = float(last.get("SUP20", np.nan))
    dist_sell = (price - sell_put) / price if price > 0 else np.nan
    tscore = trend_score(last)
    liquidity_score = bounded(avg_fin_vol / settings["full_liquidity"] if not np.isnan(avg_fin_vol) else 0)
    distance_score = bounded(dist_sell / settings["target_distance"] if not np.isnan(dist_sell) else 0)
    atr_score = bounded(1 - (atr_pct / settings["max_atr_pct"])) if not np.isnan(atr_pct) else 0.5
    support_bonus = 1.0 if not np.isnan(sup20) and sell_put <= sup20 else 0.5
    ivr = opt.get("iv_rank", np.nan)
    ivs, iv_msg = iv_score(ivr, settings["min_ivr"], settings["max_ivr"])
    premium_score = 0.5 if np.isnan(credit_width) else bounded(credit_width / 0.30)
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

    risk_per_asset = settings["risk_per_asset"]
    max_exercise_per_asset = settings["max_exercise_per_asset"]
    qty_by_loss = math.floor(risk_per_asset / (max_loss_ps * 100)) if max_loss_ps > 0 else 0
    qty_by_exercise = math.floor(max_exercise_per_asset / (sell_put * 100)) if sell_put > 0 else 0
    qty = max(0, min(qty_by_loss, qty_by_exercise, settings["max_contracts_per_asset"]))
    max_gain = (float(credit) if not np.isnan(credit) else estimated_credit) * 100 * qty
    max_loss = max_loss_ps * 100 * qty
    effective_if_exercised = sell_put - (float(credit) if not np.isnan(credit) else 0.0)
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
        "sell_price": opt.get("sell_price", np.nan),
        "buy_price": opt.get("buy_price", np.nan),
        "credit": credit,
        "credit_width": credit_width,
        "max_loss_ps": max_loss_ps,
        "iv": opt.get("iv", np.nan),
        "iv_rank": ivr,
        "volume_option": opt.get("volume_option", np.nan),
        "option_source": opt.get("source", "sem dados"),
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

# -----------------------------
# Interface Streamlit
# -----------------------------
st.set_page_config(page_title="Radar GARCH B3", page_icon="📈", layout="wide")
st.title("📈 Radar GARCH B3 — Travas de Put com Decisão Pronta")
st.caption(f"{APP_VERSION} | Venda da put no -1,5σ GARCH e compra da put no -2σ GARCH")

with st.sidebar:
    st.header("Configuração")
    margin = st.number_input("Margem/capital de referência", min_value=1000.0, value=100000.0, step=1000.0)
    max_price = st.number_input("Preço máximo do ativo", min_value=1.0, value=50.0, step=1.0)
    max_banks = st.number_input("Máximo de bancos na lista final", min_value=0, value=2, step=1)
    risk_per_asset_pct = st.slider("Risco máximo por ativo", 0.5, 10.0, 2.0, 0.5) / 100
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
    use_public = st.checkbox("Tentar dados públicos do Opções.net", value=True)
    uploaded_options = st.file_uploader("CSV opcional com dados das opções", type=["csv", "txt"])

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
}

st.info(
    "Use na sexta após o fechamento para gerar o radar da próxima semana. "
    "Na segunda após 10h30, rode novamente e confirme os prêmios no home broker antes de enviar ordem."
)

with st.expander("Formato do CSV opcional de opções"):
    st.write("O app consegue usar CSV se você exportar/copiar dados públicos. Colunas aceitas, com nomes flexíveis:")
    st.code("ticker,tipo,strike,ultimo,bid,ask,iv,iv_rank,volume,negocios,vencimento\nPETR4,PUT,46,0.32,0.30,0.34,0.38,58,100000,120,2026-05-08")

if st.button("🚀 Gerar decisão pronta", type="primary"):
    uploaded_df = parse_uploaded_options(uploaded_options)
    rows = []
    errors = []
    progress = st.progress(0)
    tickers = [t for t in selected_tickers if t in UNIVERSE]
    for i, ticker in enumerate(tickers):
        try:
            opt_df = uploaded_df.copy()
            if use_public:
                public_df = fetch_opcoesnet_tables(ticker)
                if not public_df.empty:
                    opt_df = pd.concat([opt_df, public_df], ignore_index=True) if not opt_df.empty else public_df
            row = analyze_ticker(ticker, UNIVERSE[ticker], settings, opt_df)
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
    # filtro preço e bancos na lista final
    df = df[df["price"] <= max_price].copy()
    df = df.sort_values(["decision", "score"], ascending=[True, False])
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
    display["Crédito"] = display["credit"].map(brl)
    display["Crédito/Largura"] = display["credit_width"].map(lambda x: pct(x) if not pd.isna(x) else "sem dado")
    display["IV Rank"] = display["iv_rank"].map(lambda x: num(x, 0) if not pd.isna(x) else "sem dado")
    display["ATR%"] = display["atr_pct"].map(pct)
    display["Distância Put"] = display["dist_sell"].map(pct)
    display["Qtd"] = display["qty"]
    display["Ganho Máx"] = display["max_gain"].map(brl)
    display["Perda Máx"] = display["max_loss"].map(brl)
    display["Preço efetivo se exercido"] = display["effective_if_exercised"].map(brl)

    cols = [
        "decision", "ticker", "setor", "Score", "Preço", "Vender Put", "Comprar Put",
        "Crédito", "Crédito/Largura", "IV Rank", "Qtd", "Ganho Máx", "Perda Máx",
        "Preço efetivo se exercido", "reason"
    ]
    st.dataframe(display[cols].rename(columns={
        "decision": "Decisão", "ticker": "Ativo", "setor": "Setor", "reason": "Motivo"
    }), use_container_width=True, hide_index=True)

    top_operar = final[final["decision"].eq("OPERAR SEGUNDA")]
    if not top_operar.empty:
        st.success("Há candidatos para olhar na segunda. Ainda assim, confirme vencimento, bid/ask, liquidez e prêmio no home broker.")
    else:
        st.warning("Nenhum ativo recebeu decisão automática de OPERAR. O app está sendo conservador porque faltou prêmio/IV/liquidez ou dados reais de opções.")

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
    st.write(f"**Fonte dos dados de opção:** {row['option_source']}. Se aparecer 'sem dados', use o CSV ou confira manualmente.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Banda -1,5σ", brl(row["band_minus_15"]))
    k2.metric("Banda -2σ", brl(row["band_minus_20"]))
    k3.metric("GARCH semanal", pct(row["sigma_week"]))
    k4.metric("Modelo", row["model"])

    if PLOTLY_AVAILABLE:
        try:
            chart = add_indicators(download_history(choice, settings["history_period"])).tail(140)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=chart.index, open=chart["Open"], high=chart["High"], low=chart["Low"], close=chart["Close"], name=choice))
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
    psell = s1.number_input("Prêmio real da put vendida", min_value=0.0, value=float(row["sell_price"]) if not pd.isna(row["sell_price"]) else 0.20, step=0.01)
    pbuy = s2.number_input("Prêmio real da put comprada", min_value=0.0, value=float(row["buy_price"]) if not pd.isna(row["buy_price"]) else 0.05, step=0.01)
    qty_manual = s3.number_input("Quantidade", min_value=1, value=max(int(row["qty"]), 1), step=1)
    credit = psell - pbuy
    width = float(row["width"])
    max_loss_ps = max(width - credit, 0)
    st.write(f"Crédito líquido: **{brl(credit)}** | Crédito/Largura: **{pct(credit/width if width else np.nan)}** | Perda máxima: **{brl(max_loss_ps*100*qty_manual)}** | Ganho máximo: **{brl(credit*100*qty_manual)}**")
    if credit <= 0:
        st.error("Não montar: crédito líquido zero ou negativo.")
    elif credit / width < settings["min_credit_width"]:
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
    "Sempre confira prêmios, vencimentos, bid/ask, liquidez, eventos corporativos e regras da sua corretora antes de operar."
)
