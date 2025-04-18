import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic

# ─── ページ設定 ─────────────────────────────────
st.set_page_config(page_title="TagSense", layout="wide")
st.title("🗂️ TagSense — コメントベース タグ付け＆感情分析")

# ─── サイドバー：設定 ─────────────────────────────
with st.sidebar:
    api_key = st.text_input(
        "🔑 Claude APIキー",
        type="password",
        help="Anthropicコンソールで取得したAPIキーを貼り付けてください"
    )
    max_rows = st.number_input(
        "🔢 一度に解析する最大行数",
        min_value=10, max_value=1000, value=200, step=10,
        help="大きいCSVはここで制限するとAPIエラー防止になります"
    )

# ─── CSVアップロード ────────────────────────────
uploaded = st.file_uploader(
    "📁 コメントCSVアップロード（必須列: コメント, 作成日）", type="csv"
)
if not uploaded:
    st.info("CSVをアップロードしてください")
    st.stop()

# ─── データ読み込み＆バリデーション ────────────────────
df = pd.read_csv(uploaded)
if "コメント" not in df.columns or "作成日" not in df.columns:
    st.error("CSVに必須列「コメント」「作成日」がありません")
    st.stop()
df["作成日"] = pd.to_datetime(df["作成日"], errors="coerce")

# 行数制限
if len(df) > max_rows:
    st.warning(f"解析対象を先頭{max_rows}行に制限します（全{len(df)}行中）")
    df = df.head(max_rows)

# ─── API解析関数（キャッシュ付き） ────────────────────
@st.cache_data(show_spinner=False)
def analyze_comment(comment: str, key: str):
    snippet = comment[:500]  # 最大500文字
    client = Anthropic(api_key=key)
    user_message = (
        "以下のコメント文について、関連するカテゴリタグを最大3つと、"
        "感情（ポジティブ／ネガティブ／中立）を出力してください。\n\n"
        f"{snippet}\n\n"
        "フォーマット: タグ1, タグ2, タグ3 | 感情: ラベル"
    )
    try:
        resp = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            messages=[{"role": "user", "content": user_message}],
            max_tokens=200,
            temperature=0.0,
        )
        raw = resp.content
        if isinstance(raw, list):
            raw = raw[0]
        text = raw.strip() if hasattr(raw, "strip") else str(raw)
    except Exception as e:
        # APIエラー時は空返却
        return "", ""
    # パース
    parts = text.split("|", 1)
    if len(parts) == 2:
        tags = [t.strip() for t in parts[0].split(",")][:3]
        sentiment = parts[1].replace("感情:", "").strip()
    else:
        tags, sentiment = [text], ""
    return ", ".join(tags), sentiment

# ─── 解析実行 ────────────────────────────────────
if st.button("🤖 タグ付け＆感情分析を実行"):
    if not api_key:
        st.error("APIキーを入力してください")
        st.stop()
    with st.spinner("解析中…少々お待ちください"):
        results = [analyze_comment(c, api_key) for c in df["コメント"].astype(str)]
        df["タグ"], df["感情"] = zip(*results)
    st.success("完了しました！")

# ─── 結果表示＆ダッシュボード ────────────────────
if "タグ" in df.columns:
    st.subheader("📋 結果一覧")
    st.dataframe(df)

    st.subheader("📊 ダッシュボード")
    # タグ件数
    tag_counts = (
        df["タグ"].str.split(",", expand=True)
        .stack().str.strip().value_counts()
    )
    fig1, ax1 = plt.subplots()
    tag_counts.plot.bar(ax=ax1)
    ax1.set_ylabel("件数")
    st.pyplot(fig1)

    # 感情分布
    sent_counts = df["感情"].fillna("未分類").value_counts()
    fig2, ax2 = plt.subplots()
    sent_counts.plot.pie(autopct="%1.1f%%", ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)

    # 時系列
    st.markdown("**時系列トレンド（週次）**")
    ts = df.set_index("作成日").resample("W").size()
    st.line_chart(ts)

    # DLボタン
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 CSVダウンロード", csv_data, "tagged_results.csv", mime="text/csv"
    )
else:
    st.info("「タグ付け＆感情分析を実行」ボタンを押してください")
