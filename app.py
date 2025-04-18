import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from anthropic import Anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime

# ─── 定数設定 ─────────────────────────────────────────
BATCH_SIZE  = 20      # 1リクエストあたりのコメント件数
MAX_WORKERS = 4       # 並列スレッド数
MODEL_NAME  = "claude-3-7-sonnet-20250219"

# ─── ページ設定 ─────────────────────────────────────────
st.set_page_config(page_title="TagSense", layout="wide")
st.title("🗂️ TagSense — 自動タグ付けダッシュボード")

# ─── サイドバー：APIキー入力 ─────────────────────────────
api_key = st.sidebar.text_input(
    "🔑 Claude APIキーを入力",
    type="password",
    help="Anthropicコンソールで取得したAPIキーを貼り付けてください"
)
if not api_key:
    st.sidebar.warning("APIキーがないとタグ付けできません")

# ─── CSVアップロード ─────────────────────────────────────
uploaded = st.file_uploader(
    "📁 コメントCSVをアップロード（必須列: コメント, 作成日）",
    type="csv"
)
if not uploaded:
    st.info("まずはCSVファイルをアップロードしてください")
    st.stop()

# ─── データ読み込み＆バリデーション ─────────────────────
df = pd.read_csv(uploaded)
if "コメント" not in df.columns or "作成日" not in df.columns:
    st.error("CSVに必須列「コメント」「作成日」がありません")
    st.stop()
df["作成日"] = pd.to_datetime(df["作成日"], errors="coerce")

# ─── バッチ解析関数 ─────────────────────────────────────
def analyze_batch(comments: list[str], key: str) -> list[tuple[str, str]]:
    # 各コメントを先頭200文字に切って改行除去
    snippets = [str(c)[:200].replace("\n", " ") for c in comments]
    prompt = "以下のコメントリストについて、番号付きで「メインタグ」と「サブタグ（最大2つ）」を出力してください。\n\n"
    for i, s in enumerate(snippets, 1):
        prompt += f"{i}. {s}\n"
    prompt += "\n出力フォーマット:\n1. メインタグ: <タグ> | サブタグ: <タグ1>, <タグ2>\n…\n"

    client = Anthropic(api_key=key)
    try:
        resp = client.messages.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        raw = resp.content
        if isinstance(raw, list) and raw:
            raw = raw[0]
        text = raw.strip() if hasattr(raw, "strip") else str(raw)
    except Exception:
        # API 呼び出し失敗時はすべて空タグで埋める
        return [("", "") for _ in comments]

    lines = [line for line in text.splitlines() if line.strip()]
    results: list[tuple[str, str]] = []
    for line in lines[: len(comments)]:
        # "1. メインタグ: xxx | サブタグ: yyy, zzz"
        try:
            _, rest = line.split(".", 1)
            main_part, sub_part = rest.split("|", 1)
            main = main_part.split(":", 1)[1].strip()
            subs = [t.strip() for t in sub_part.split(":", 1)[1].split(",")][:2]
            sub = ", ".join(subs)
        except Exception:
            # パース失敗時は空文字
            main, sub = "", ""
        results.append((main, sub))
    # 足りない分は空文字で埋める
    while len(results) < len(comments):
        results.append(("", ""))
    return results

# ─── 並列バッチ実行 ───────────────────────────────────
def analyze_all(comments: list[str], key: str) -> list[tuple[str, str]]:
    total = len(comments)
    out: list[tuple[str, str]] = [("", "")] * total
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
            except Exception:
                # バッチ失敗時は空タグ
                for i in range(start, min(start + BATCH_SIZE, total)):
                    out[i] = ("", "")
    return out

# ─── タグ付け実行 ─────────────────────────────────────
if st.button("🤖 タグ付けを実行"):
    if not api_key:
        st.error("先にAPIキーを入力してください")
        st.stop()
    comments = df["コメント"].astype(str).tolist()
    with st.spinner("解析中…しばらくお待ちください"):
        pairs = analyze_all(comments, api_key)
        df["メインタグ"], df["サブタグ"] = zip(*pairs)
    st.success("✅ タグ付けが完了しました！")

# ─── ダッシュボード＆CSVダウンロード ────────────────────
if "メインタグ" in df.columns:
    # KPI
    total      = len(df)
    main_counts = df["メインタグ"].value_counts()
    unique_main = main_counts.size
    top_main    = int(main_counts.iloc[0]) if not main_counts.empty else 0

    k1, k2, k3 = st.columns(3)
    k1.metric("総コメント数", f"{total}")
    k2.metric("ユニークメインタグ数", f"{unique_main}")
    k3.metric("トップメインタグ件数", f"{top_main}")

    st.markdown("---")

    # メインタグランキング
    st.subheader("メインタグ 件数ランキング")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    main_counts.plot.bar(ax=ax1)
    ax1.set_ylabel("件数")
    st.pyplot(fig1, use_container_width=True)

    # 週次トレンド（メインタグ別）
    st.subheader("メインタグ 週次トレンド")
    weekly = (
        df.set_index("作成日")["メインタグ"]
          .groupby(pd.Grouper(freq="W"))
          .value_counts()
          .unstack(fill_value=0)
    )
    st.line_chart(weekly, use_container_width=True)

    # 新規メインタグ検出
    first = df["作成日"].min()
    baseline_end = first + datetime.timedelta(days=6)
    baseline_tags = set(df[df["作成日"] <= baseline_end]["メインタグ"])
    all_tags = set(df["メインタグ"])
    new_tags = sorted(all_tags - baseline_tags)
    if new_tags:
        st.subheader("過去未出メインタグ")
        for t in new_tags:
            st.write(f"- {t}")

    st.markdown("---")

    # CSVダウンロード
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 タグ付きCSVをダウンロード",
        csv_data,
        "tagged_results.csv",
        mime="text/csv"
    )
else:
    st.info("タグ付けを実行するとダッシュボードが表示されます")
