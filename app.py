import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from anthropic._utils._errors import BadRequestError

# ─── ページ設定 ─────────────────────────────────
st.set_page_config(page_title="TagSense", layout="wide")
st.title("🗂️ TagSense — コメントベース 自動タグ付け＆感情分析")

# ─── サイドバー：APIキー入力 ─────────────────────
api_key = st.sidebar.text_input(
    "🔑 Claude APIキーを入力", type="password",
    help="Anthropicコンソールで取得したAPIキーを貼り付けてください"
)

# ─── CSVアップロード ────────────────────────────
uploaded = st.file_uploader(
    "📁 コメントCSVをアップロード（必須列: コメント, 作成日）", type="csv"
)
if not uploaded:
    st.info("CSVファイルをアップロードしてください")
    st.stop()

# ─── データ読み込みと日付変換 ────────────────────
df = pd.read_csv(uploaded)
for col in ("コメント", "作成日"):
    if col not in df.columns:
        st.error(f"CSVに必須列「{col}」がありません。列名を確認してください。")
        st.stop()
df["作成日"] = pd.to_datetime(df["作成日"], errors="coerce")

# ─── Claude呼び出し関数 ──────────────────────────
def analyze_with_claude(comment: str, api_key: str):
    # 長いコメントは先頭500文字にトリミング
    snippet = comment[:500]
    client = Anthropic(api_key=api_key)
    prompt = (
        f"{HUMAN_PROMPT}"
        "以下のコメント文に対して、関連するカテゴリタグを最大3つと、"
        "感情（ポジティブ／ネガティブ／中立）を出力してください。\n\n"
        f"{snippet}\n\n"
        "フォーマット：タグ1, タグ2, タグ3 | 感情: ラベル\n"
        f"{AI_PROMPT}"
    )
    try:
        resp = client.completions.create(
            model="claude-3-7-sonnet-20250219",
            prompt=prompt,
            max_tokens_to_sample=100,
            temperature=0.0,
        )
    except BadRequestError as e:
        # エラー詳細を画面表示＆ログにも出力
        st.error(f"APIリクエストでエラー: {e}")
        st.stop()
    result = resp.completion.strip()
    try:
        tags_part, sent_part = result.split("|")
        tags = [t.strip() for t in tags_part.split(",")][:3]
        sentiment = sent_part.replace("感情:", "").strip()
    except Exception:
        tags, sentiment = [result], ""
    return ", ".join(tags), sentiment

# ─── 実行ボタン ───────────────────────────────────
if st.button("🤖 タグ付け＆感情分析を実行"):
    if not api_key:
        st.error("APIキーを入力してください")
        st.stop()
    with st.spinner("🛠️ Claudeで解析中…しばらくお待ちください"):
        df["タグ"], df["感情"] = zip(*[
            analyze_with_claude(txt, api_key)
            for txt in df["コメント"].astype(str)
        ])
    st.success("✅ 完了しました！")

# ─── 結果表示＆ダッシュボード ────────────────────
if "タグ" in df.columns:
    st.subheader("📋 タグ＆感情付き結果一覧")
    st.dataframe(df[["コメントID","コメント","タグ","感情","作成日"]])

    st.subheader("📊 ダッシュボード")
    # カテゴリ別 件数ランキング
    tag_counts = df["タグ"].str.split(",", expand=True).stack().str.strip().value_counts()
    fig1, ax1 = plt.subplots()
    tag_counts.plot.bar(ax=ax1); ax1.set_ylabel("件数")
    st.pyplot(fig1)
    # 感情分布
    sent_counts = df["感情"].fillna("未分類").value_counts()
    fig2, ax2 = plt.subplots()
    sent_counts.plot.pie(autopct="%1.1f%%", ax=ax2); ax2.set_ylabel("")
    st.pyplot(fig2)
    # 時系列トレンド（週次）
    st.markdown("**時系列トレンド（週次）**")
    ts = df.set_index("作成日").resample("W").size()
    st.line_chart(ts)
    # CSVダウンロード
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSVダウンロード", csv_data, "tagged_results.csv", mime="text/csv")
else:
    st.info("“タグ付け＆感情分析を実行”ボタンを押してください")
