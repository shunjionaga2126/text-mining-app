import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# ─── UI設定 ─────────────────────────────────────
st.set_page_config(page_title="TagSense", layout="wide")
st.title("🗂️ TagSense — Claudeで自動タグ付け＆ダッシュボード")

# ─── セッション状態 ─────────────────────────────
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# ─── APIキー入力 ───────────────────────────────
with st.expander("🔑 Claude APIキーを入力"):
    key = st.text_input(
        "APIキー",
        type="password",
        value=st.session_state.api_key,
        help="Anthropicコンソールで取得したAPIキーを貼り付けてください"
    )
    if key:
        st.session_state.api_key = key

# ─── CSVアップロード ────────────────────────────
uploaded = st.file_uploader("📁 コールログCSVをアップロード", type="csv")
if not uploaded:
    st.info("まずはCSVファイルをアップロードしてください")
    st.stop()

# DataFrame読み込み
df = pd.read_csv(uploaded)

# ─── 列選択 ────────────────────────────────────
cols = df.columns.tolist()
default_text = "text" if "text" in cols else cols[0]
text_col = st.selectbox("▶ 本文が入っている列を選択してください", cols, index=cols.index(default_text))
date_col = None
if "date" in cols:
    date_col = "date"
else:
    choice = st.selectbox("▶ 日付列を選択（スキップ可）", ["(なし)"] + cols)
    if choice != "(なし)":
        date_col = choice

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ─── Claude連携関数 ────────────────────────────
def analyze_with_claude(text: str, api_key: str) -> tuple[str, str]:
    """
    問い合わせ文に対して、
    ・カテゴリタグ(最大3つ)
    ・感情（ポジティブ／ネガティブ／中立）
    を返す
    """
    client = Anthropic(api_key=api_key)
    prompt = (
        f"{HUMAN_PROMPT}"
        "以下の問い合わせ文について、関連するカテゴリタグを最大3つと、"
        "感情（ポジティブ／ネガティブ／中立）を出力してください。\n\n"
        f"{text}\n\n"
        "フォーマット：タグ1, タグ2, タグ3 | 感情: ラベル\n"
        f"{AI_PROMPT}"
    )
    resp =
