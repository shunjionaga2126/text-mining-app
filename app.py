# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sudachipy import dictionary, tokenizer
import json

# ─── サイドバー設定 ─────────────────────────
st.sidebar.title("🔧 設定")
uploaded = st.sidebar.file_uploader("📁 コールログCSVアップロード", type="csv")

# 辞書＆除外ワード永続化ファイル
DICT_FILE = "keywords.json"
try:
    with open(DICT_FILE, "r", encoding="utf-8") as f:
        dic = json.load(f)
except FileNotFoundError:
    dic = {"include": [], "exclude": []}

# 辞書登録
new_inc = st.sidebar.text_input("⭐️ 辞書に追加")
if st.sidebar.button("辞書登録"):
    if new_inc and new_inc not in dic["include"]:
        dic["include"].append(new_inc)
        with open(DICT_FILE, "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=2)
# 除外ワード登録
new_exc = st.sidebar.text_input("🚫 除外ワード追加")
if st.sidebar.button("除外登録"):
    if new_exc and new_exc not in dic["exclude"]:
        dic["exclude"].append(new_exc)
        with open(DICT_FILE, "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=2)

# ─── メイン画面 ───────────────────────────────
st.title("🔍 BPO向けテキストマイニングデモ")

if not uploaded:
    st.info("まずはサイドバーでCSVをアップしてください")
    st.stop()

# CSV読み込み＋バリデーション
df = pd.read_csv(uploaded)
if "コメント" not in df.columns or "作成日" not in df.columns:
    st.error("CSVに「コメント」「作成日」列が必要です")
    st.stop()
df["作成日"] = pd.to_datetime(df["作成日"], errors="coerce")
df = df.dropna(subset=["コメント","作成日"]).drop_duplicates(subset=["コメント","作成日"])

# 集計期間フィルタ
st.subheader("📅 集計期間フィルタ")
dmin, dmax = st.date_input("期間を選択", [df["作成日"].min(), df["作成日"].max()])
mask = (df["作成日"] >= pd.to_datetime(dmin)) & (df["作成日"] <= pd.to_datetime(dmax))
df = df[mask]

# 形態素解析＆頻出キーワード抽出
st.subheader("📊 頻出キーワードランキング")
tokenizer_obj = dictionary.Dictionary().create()
STOP = set(["は","の","が","を","に","で","と","も","た","です","ます"])
ctr = Counter()
for text in df["コメント"]:
    for m in tokenizer_obj.tokenize(str(text), tokenizer.Tokenizer.SplitMode.C):
        w = m.surface()
        if len(w) > 1 and w not in STOP:
            ctr[w] += 1
kw = ctr.most_common(30)
# 辞書優先＆除外フィルタ
kw = [(w,c) for w,c in kw if w not in dic["exclude"]]
for w in dic["include"]:
    if w in dict(kw):
        kw.insert(0,(w,dict(kw)[w]))
words, counts = zip(*kw) if kw else ([],[])
fig, ax = plt.subplots(figsize=(8,4))
ax.barh(words, counts)
ax.invert_yaxis()
ax.set_xlabel("出現回数")
st.pyplot(fig, use_container_width=True)

# キーワード選択→該当全文 or DL
st.subheader("🔍 キーワードで全文表示 / CSVダウンロード")
sel = st.selectbox("キーワードを選択", words)
if sel:
    subdf = df[df["コメント"].str.contains(sel, na=False)]
    st.write(subdf)
    csv_data = subdf.to_csv(index=False).encode("utf-8")
    st.download_button("📥 フィルタ結果DL", csv_data, f"{sel}_results.csv")

# 辞書・除外ワード一覧
st.sidebar.markdown("---")
st.sidebar.write("Current 辞書ワード:", dic["include"])
st.sidebar.write("Current 除外ワード:", dic["exclude"])
