import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# ─── 定数設定 ─────────────────────────────────────
BATCH_SIZE = 20      # 1回のAPI呼び出しでまとめて送るコメント数
MAX_WORKERS = 4      # 並列で動かすスレッド数

# ─── ページ設定 ─────────────────────────────────
st.set_page_config(page_title="TagSense", layout="wide")
st.title("🗂️ TagSense — 自動タグ付けダッシュボード")

# ─── サイドバー：APIキー入力 ─────────────────────
api_key = st.sidebar.text_input(
    "🔑 Claude APIキーを入力",
    type="password",
    help="Anthropicコンソールで取得したAPIキーを貼り付けてください"
)
if not api_key:
    st.sidebar.info("APIキーがないと解析できません")

# ─── CSVアップロード ────────────────────────────
uploaded = st.file_uploader(
    "📁 コメントCSVをアップロード（必須列: コメント, 作成日）",
    type="csv"
)
if not uploaded:
    st.info("まずはCSVファイルをアップロードしてください")
    st.stop()

# ─── データ読み込み＆バリデーション ───────────────────
df = pd.read_csv(uploaded)
if "コメント" not in df.columns or "作成日" not in df.columns:
    st.error("CSVに必須列「コメント」「作成日」がありません")
    st.stop()
df["作成日"] = pd.to_datetime(df["作成日"], errors="coerce")

# ─── バッチ解析関数 ─────────────────────────────
def analyze_batch(comments: list[str], key: str) -> list[tuple[str,str]]:
    """ BATCH_SIZE 件まとめてAPIコールし、(tags, sentiment) を返す """
    snippets = [str(c)[:200].replace("\n", " ") for c in comments]
    prompt = "以下のコメントリストについて、番号付きで最大3つのカテゴリタグを出力してください。\n\n"
    for i, s in enumerate(snippets, 1):
        prompt += f"{i}. {s}\n"
    prompt += "\n出力フォーマット:\n1. タグ1,タグ2,タグ3\n…\n"
    client = Anthropic(api_key=key)
    resp = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.0,
    )
    raw = resp.content
    if isinstance(raw, list) and raw:
        raw = raw[0]
    text = raw.strip() if hasattr(raw, "strip") else str(raw)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    results = []
    for line in lines[: len(comments)]:
        try:
            _, rest = line.split(".", 1)
            tags = [t.strip() for t in rest.split(",")][:3]
        except:
            tags = [line.strip()]
        results.append(( ", ".join(tags), "" ))
    while len(results) < len(comments):
        results.append(("", ""))
    return results

# ─── 並列バッチ実行 ────────────────────────────
def analyze_all(comments: list[str], key: str) -> list[tuple[str,str]]:
    total = len(comments)
    out: list[tuple[str,str]] = [("", "")] * total
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {}
        for start in range(0, total, BATCH_SIZE):
            batch = comments[start : start + BATCH_SIZE]
            fut = exe.submit(analyze_batch, batch, key)
            futures[fut] = start
        for fut in as_completed(futures):
            start = futures[fut]
            try:
                batch_res = fut.result()
                for i, res in enumerate(batch_res):
                    out[start + i] = res
            except:
                for i in range(start, min(start + BATCH_SIZE, total)):
                    out[i] = ("", "")
    return out

# ─── 実行ボタン ─────────────────────────────────
if st.button("🤖 タグ付けを実行"):
    if not api_key:
        st.error("APIキーを入力してください")
        st.stop()
    comments = df["コメント"].astype(str).tolist()
    with st.spinner("解析中…しばらくお待ちください"):
        tags = [pair[0] for pair in analyze_all(comments, api_key)]
        df["タグ"] = tags
    st.success("✅ タグ付けが完了しました！")

# ─── ダッシュボード表示 ────────────────────────────
if "タグ" in df.columns:
    st.subheader("📊 タグ別 件数ランキング")
    tag_counts = (
        df["タグ"]
        .str.split(",", expand=True)
        .stack()
        .str.strip()
        .value_counts()
    )
    st.bar_chart(tag_counts)
    st.markdown("---")
    st.download_button(
        "📥 タグ付きCSVをダウンロード",
        df.to_csv(index=False).encode("utf-8"),
        "tagged_results.csv",
        mime="text/csv"
    )
else:
    st.info("タグ付けを実行するとグラフが表示されます")
