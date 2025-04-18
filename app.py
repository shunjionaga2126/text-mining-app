import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# ─── ページ設定 ─────────────────────────────────
st.set_page_config(page_title="TagSense", layout="wide")
st.title("🗂️ TagSense — コメントベース タグ付け＆感情分析")

# ─── サイドバー：APIキー入力 ─────────────────────
api_key = st.sidebar.text_input(
    "🔑 Claude APIキーを入力",
    type="password",
    help="Anthropicコンソールで取得したAPIキーを貼り付けてください"
)

# ─── CSVアップロード ────────────────────────────
uploaded = st.file_uploader(
    "📁 CSVをアップロード（必須列: チケットID, コメントID, コメント, 作成日）",
    type="csv"
)
if not uploaded:
    st.info("まずはCSVファイルをアップロードしてください")
    st.stop()

# ─── データ読み込み＆日付変換 ────────────────────
df = pd.read_csv(uploaded)
required = ["チケットID", "コメントID", "コメント", "作成日"]
for col in required:
    if col not in df.columns:
        st.error(f"CSVに必須列「{col}」がありません。列名を確認してください。")
        st.stop()
df["作成日"] = pd.to_datetime(df["作成日"], errors="coerce")

# ─── Claude Messages API 呼び出し関数 ─────────────────
def analyze_with_claude(comment: str, api_key: str):
    snippet = comment[:500]  # 長いコメントは先頭500文字に制限
    client = Anthropic(api_key=api_key)
    # Messages APIでは messages に HUMAN_PROMPT/AI_PROMPT を含めず、
    # user-role のみ指定します
    user_message = (
        f"{HUMAN_PROMPT}"
        "以下のコメント文について、関連するカテゴリタグを最大3つと、"
        "感情（ポジティブ／ネガティブ／中立）を出力してください。\n\n"
        f"{snippet}\n\n"
        "フォーマット: タグ1, タグ2, タグ3 | 感情: ラベル"
    )
    try:
        resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": user_message}],
            max_tokens_to_sample=100,
            temperature=0.0,
            thinking={"type": "enabled", "budget_tokens": 1600},
        )
        result = resp.content.strip()
    except Exception as e:
        st.error(f"APIエラー: {e}")
        return "", ""
    # 「|」でタグと感情に分割
    try:
        tags_part, sent_part = result.split("|")
        tags = [t.strip() for t in tags_part.split(",")][:3]
        sentiment = sent_part.replace("感情:", "").strip()
    except Exception:
        tags, sentiment = [result], ""
    return ", ".join(tags), sentiment

# ─── タグ付け＆感情分析 実行 ────────────────────
if st.button("🤖 タグ付け＆感情分析を実行"):
    if not api_key:
        st.error("先にAPIキーを入力してください")
        st.stop()
    with st.spinner("🛠️ 解析中…しばらくお待ちください"):
        df["タグ"], df["感情"] = zip(*[
            analyze_with_claude(c, api_key)
            for c in df["コメント"].astype(str)
        ])
    st.success("✅ 完了しました！")

# ─── 結果表示＆ダッシュボード ────────────────────
if "タグ" in df.columns:
    st.subheader("📋 タグ＆感情付き結果一覧")
    st.dataframe(df[["チケットID", "コメントID", "コメント", "タグ", "感情", "作成日"]])

    st.subheader("📊 ダッシュボード")

    # ■ カテゴリ別 件数ランキング
    tag_counts = (
        df["タグ"].str.split(",", expand=True)
        .stack().str.strip().value_counts()
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

    # ■ 時系列トレンド（週次集計）
    ts = df.set_index("作成日").resample("W").size()
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
    st.info("「🤖 タグ付け＆感情分析を実行」ボタンを押してください")
