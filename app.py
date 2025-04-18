import re
import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic

# ─── 定数 ─────────────────────────────────────────
MODEL_NAME  = "claude-3-7-sonnet-20250219"
BATCH_SIZE  = 20     # 一度に投げるコメント数
CACHE_TTL   = 3600   # キャッシュ有効秒

# ─── ページ設定 ───────────────────────────────────
st.set_page_config(page_title="TagSense", layout="wide")
st.title("🗂️ TagSense — メインタグ＆サブタグ 自動生成ダッシュボード")

# ─── サイドバー ───────────────────────────────────
with st.sidebar:
    api_key = st.text_input("🔑 Claude APIキー", type="password")
    st.markdown("---")
    st.write("バッチ設定")
    bsize = st.slider("一度に投げるコメント数", 5, 50, BATCH_SIZE, 5)
    st.write("キャッシュTTL (秒)")
    ttl = st.number_input("", 300, 86400, CACHE_TTL, 300)
if not api_key:
    st.sidebar.error("APIキーを入力してください")

# ─── CSVアップロード ─────────────────────────────────
uploaded = st.file_uploader(
    "📁 コメントCSVをアップロード（必須列: コメント, 作成日）", type="csv"
)
if not uploaded:
    st.info("CSVをアップロードして下さい")
    st.stop()

# ─── データ読み込み＆検証 ───────────────────────────
df = pd.read_csv(uploaded)
if "コメント" not in df.columns or "作成日" not in df.columns:
    st.error("CSVに「コメント」「作成日」列が必要です")
    st.stop()
df["作成日"] = pd.to_datetime(df["作成日"], errors="coerce")

# ─── LLM呼び出し（バッチ） ─────────────────────────
@st.cache_data(ttl=ttl)
def fetch_tags(snippets: tuple[str], key: str) -> list[tuple[str,str]]:
    # プロンプト作成
    prompt = "以下のコメントについて、番号付きで【メインタグ】と【サブタグ（最大2つ）】を出力してください。\n\n"
    for i, s in enumerate(snippets, 1):
        prompt += f"{i}. {s[:200].replace(chr(10),' ')}\n"
    prompt += "\n出力フォーマット:\n1. メインタグ: <タグ> | サブタグ: <タグ1>, <タグ2>\n…\n"

    client = Anthropic(api_key=key)
    try:
        resp = client.messages.create(
            model=MODEL_NAME,
            messages=[{"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.0,
        )
    except Exception:
        return [("", "") for _ in snippets]

    raw = resp.content
    if isinstance(raw, list):
        raw = raw[0]
    text = raw.strip() if hasattr(raw, "strip") else str(raw)

    # 「1.」「2.」…で分割
    parts = re.split(r"\d+\.\s", text)
    # 最初の空要素を除き、必要分だけ使う
    entries = [p.strip() for p in parts if p.strip()][: len(snippets)]

    results = []
    for entry in entries:
        # entry例: "メインタグ: 決済 | サブタグ: クレジット, 店頭受取"
        try:
            main_part, sub_part = entry.split("|", 1)
            main = main_part.split(":",1)[1].strip()
            subs = [t.strip() for t in sub_part.split(":",1)[1].split(",")][:2]
            sub = ", ".join(subs)
        except Exception:
            main, sub = "", ""
        results.append((main, sub))
    # 足りない分は空で埋める
    results += [("", "")] * (len(snippets) - len(results))
    return results

# ─── 全コメント解析 ─────────────────────────────────
def generate_tags(comments: list[str], key: str, batch_size: int):
    mains, subs = [], []
    for i in range(0, len(comments), batch_size):
        batch = tuple(comments[i:i+batch_size])
        pairs = fetch_tags(batch, key)
        for m, s in pairs:
            mains.append(m)
            subs.append(s)
    return pd.DataFrame({"メインタグ": mains, "サブタグ": subs})

# ─── タグ付け実行 ─────────────────────────────────
if st.button("🤖 タグ付けを実行"):
    if not api_key:
        st.error("APIキーが必要です")
        st.stop()
    comments = df["コメント"].astype(str).tolist()
    with st.spinner("タグを生成中…お待ちください"):
        tag_df = generate_tags(comments, api_key, bsize)
        df[["メインタグ","サブタグ"]] = tag_df
    st.success("✅ タグ付け完了！")

# ─── ダッシュボード＆CSVダウンロード ────────────────────
if "メインタグ" in df.columns:
    # KPI
    total       = len(df)
    main_counts = df["メインタグ"].value_counts()
    k1,k2,k3    = st.columns(3)
    k1.metric("コメント件数", total)
    k2.metric("ユニークメインタグ", main_counts.size)
    k3.metric("トップタグ件数", int(main_counts.iloc[0]) if not main_counts.empty else 0)

    st.markdown("---")
    # メインタグ分布
    st.subheader("メインタグ 分布")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    main_counts.plot.bar(ax=ax1); ax1.set_ylabel("件数")
    st.pyplot(fig1, use_container_width=True)

    # 週次トレンド
    st.subheader("週次トレンド（メインタグ）")
    trend = (
        df.set_index("作成日")["メインタグ"]
          .groupby(pd.Grouper(freq="W"))
          .value_counts()
          .unstack(fill_value=0)
    )
    st.line_chart(trend, use_container_width=True)

    # 新規タグ検出
    first = df["作成日"].min()
    baseline = set(df[df["作成日"] <= first + datetime.timedelta(days=6)]["メインタグ"])
    new_tags = sorted(set(df["メインタグ"]) - baseline)
    if new_tags:
        st.subheader("過去未出メインタグ")
        for tag in new_tags:
            st.write(f"- {tag}")

    st.markdown("---")
    # CSVダウンロード
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 タグ付きCSVダウンロード", csv, "tagged_results.csv", mime="text/csv")
else:
    st.info("「タグ付けを実行」後にダッシュボードが表示されます")
