import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# ─── ページ設定 ─────────────────────────────────
st.set_page_config(page_title="TagSense", layout="wide")
st.title("🗂️ TagSense — Claudeで自動タグ付け＆ダッシュボード")

# ─── サイドバー：APIキー入力 ─────────────────────
api_key = st.sidebar.text_input(
    "🔑 Claude APIキーを入力",
    type="password",
    help="Anthropicコンソールで取得したAPIキーを貼り付けてください"
)

# ─── CSVアップロード ────────────────────────────
uploaded = st.file_uploader("📁 コールログCSVをアップロード（必須列: text, date）", type="csv")
if not uploaded:
    st.info("まずはCSVファイルをアップロードしてください")
    st.stop()

# ─── データ読み込み・日付変換 ────────────────────
df = pd.read_csv(uploaded)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ─── Claude連携関数 ────────────────────────────
def analyze_with_claude(text: str, api_key: str):
    """
    text に対して
      ・最大3つのカテゴリタグ
      ・感情（ポジティブ／ネガティブ／中立）
    を返します（タプル: (tags, sentiment)）。
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
        model="claude-3-7-sonnet-20250219",
        prompt=prompt,
        max_tokens_to_sample=100,
        temperature=0.0,
    )
    result = resp.completion.strip()
    # 解析結果をパース
    try:
        tags_part, sent_part = result.split("|")
        tags = [t.strip() for t in tags_part.split(",")][:3]
        sentiment = sent_part.replace("感情:", "").strip()
    except Exception:
        tags = [result]
        sentiment = ""
    return ", ".join(tags), sentiment

# ─── タグ付け＆感情分析実行 ────────────────────
if st.button("🤖 タグ付け＆感情分析を実行"):
    if not api_key:
        st.error("APIキーを入力してください")
        st.stop()
    with st.spinner("🛠️ Claudeで解析中…少々お待ちください"):
        df["タグ"], df["感情"] = zip(
            *df["text"].astype(str).apply(lambda x: analyze_with_claude(x, api_key))
        )
    st.success("完了しました！")

# ─── 結果表示＆ダッシュボード ────────────────────
if "タグ" in df.columns:
    st.subheader("📋 タグ付き結果一覧")
    st.dataframe(df)

    st.subheader("📊 ダッシュボード")

    # ■ カテゴリ別 件数ランキング
    tag_counts = (
        df["タグ"]
        .str.split(",", expand=True)
        .stack()
        .str.strip()
        .value_counts()
    )
    fig1, ax1 = plt.subplots()
    tag_counts.plot.bar(ax=ax1)
    ax1.set_ylabel("件数")
    st.pyplot(fig1)

    # ■ 感情分布
    sent_counts = df["感情"].fillna("未分類").value_counts()
    fig2, ax2 = plt.subplots()
    sent_counts.plot.pie(autopct="%1.1f%%", ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)

    # ■ 時系列トレンド（週次）
    if "date" in df.columns:
        ts = df.set_index("date").resample("W").size()
        st.line_chart(ts)

    # ■ CSVダウンロード
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 分析結果をCSVダウンロード",
        csv_data,
        "tagged_results.csv",
        mime="text/csv"
    )
else:
    st.info("“🤖 タグ付け＆感情分析を実行” ボタンを押してください")
