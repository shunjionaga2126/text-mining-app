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
# 本文列
default_text = "text" if "text" in cols else cols[0]
text_col = st.selectbox("▶ 本文が入っている列を選択してください", cols, index=cols.index(default_text))
# 日付列（任意）
date_col = None
if "date" in cols:
    date_col = "date"
else:
    choice = st.selectbox("▶ 日付が入っている列を選択（スキップ可）", ["(なし)"] + cols)
    if choice != "(なし)":
        date_col = choice

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# ─── Claude連携関数 ────────────────────────────
def analyze_with_claude(text: str, api_key: str) -> tuple[str,str]:
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
    resp = client.completions.create(
        model="claude-3-sonnet-20240229",
        prompt=prompt,
        max_tokens_to_sample=100,   # ← 修正ポイント
        temperature=0.0,
    )
    result = resp.completion.strip()
    try:
        tags_part, sentiment_part = result.split("|")
        tags = [t.strip() for t in tags_part.split(",")][:3]
        sentiment = sentiment_part.replace("感情:", "").strip()
    except Exception:
        tags = [result]
        sentiment = ""
    return ", ".join(tags), sentiment

# ─── タグ付け実行 ─────────────────────────────
if st.button("🤖 タグ付け＆感情分析を実行"):
    if not st.session_state.api_key:
        st.error("先にAPIキーを入力してください")
        st.stop()
    with st.spinner("🛠️ Claudeで解析中…少々お待ちください"):
        df["タグ"], df["感情"] = zip(
            *df[text_col].astype(str).apply(
                lambda x: analyze_with_claude(x, st.session_state.api_key)
            )
        )
    st.success("完了しました！")

# ─── 結果表示＆ダッシュボード ────────────────────
if "タグ" in df.columns:
    st.subheader("📋 タグ付き結果一覧")
    st.dataframe(df)

    st.subheader("📊 ダッシュボード")

    # カテゴリ件数ランキング
    tag_counts = (
        df["タグ"]
        .str.split(",", expand=True)
        .stack()
        .str.strip()
        .value_counts()
    )
    st.markdown("**カテゴリ別 件数ランキング**")
    fig1, ax1 = plt.subplots()
    tag_counts.plot.bar(ax=ax1)
    ax1.set_ylabel("件数")
    st.pyplot(fig1)

    # 感情分布
    sent_counts = df["感情"].fillna("未分類").value_counts()
    st.markdown("**感情分布**")
    fig2, ax2 = plt.subplots()
    sent_counts.plot.pie(autopct="%1.1f%%", ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)

    # 時系列トレンド
    if date_col:
        st.markdown("**時系列トレンド（全件）**")
        ts = df.set_index(date_col).resample("W").size()
        st.line_chart(ts)

    # CSVダウンロード
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 分析結果をCSVダウンロード", csv, "tagged_results.csv", mime="text/csv")
else:
    st.info("“🤖 タグ付け＆感情分析を実行” ボタンを押してください")
