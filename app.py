# -*- coding: utf-8 -*-
"""
=======================================================================================
 CAFFE BENE — Өдрийн тооллого, Борлуулалтын систем тулгалтын веб апп
=======================================================================================
Ашиглах сангууд:
    pip install streamlit pandas numpy fuzzywuzzy python-Levenshtein openpyxl pytesseract Pillow

Зургаас (screenshot) уншуулах OCR боломж ажиллахын тулд Tesseract OCR систем дээр
суусан байх шаардлагатай (packages.txt файлд tesseract-ocr, tesseract-ocr-mon орсон
байх ёстой — Streamlit Cloud дээр автоматаар суулгана).

Ажиллуулах:
    streamlit run app.py
=======================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import re
import uuid
from datetime import datetime, date

try:
    from fuzzywuzzy import fuzz, process
except ImportError:
    st.error("fuzzywuzzy сан суугаагүй байна. Терминал дээр: pip install fuzzywuzzy python-Levenshtein")
    st.stop()

try:
    import pytesseract
    from PIL import Image as PILImage
    OCR_LIBS_AVAILABLE = True
except ImportError:
    OCR_LIBS_AVAILABLE = False


# =======================================================================================
# 0. ЕРӨНХИЙ ТОХИРГОО
# =======================================================================================
st.set_page_config(
    page_title="Caffe Bene | Тооллого & Тулгалт",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "cafe_bene_data"
os.makedirs(DATA_DIR, exist_ok=True)

PATH_MASTER = os.path.join(DATA_DIR, "master_items.json")
PATH_CURRENT = os.path.join(DATA_DIR, "inventory_current.json")
PATH_HISTORY = os.path.join(DATA_DIR, "inventory_history.json")
PATH_DELETED = os.path.join(DATA_DIR, "inventory_deleted.json")
PATH_PHOTOS = os.path.join(DATA_DIR, "photos")
os.makedirs(PATH_PHOTOS, exist_ok=True)

FUZZY_THRESHOLD = 70  # Нэрээр тулгах босго оноо (0-100)

# ---- Responsive / хөнгөн загвар (CSS) ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Noto Sans', sans-serif; }
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
    .cb-header {
        background: linear-gradient(90deg,#4b2e19,#7a4a24);
        padding: 18px 22px; border-radius: 14px; margin-bottom: 14px;
        color: white;
    }
    .cb-header h1 {margin:0; font-size: 1.5rem;}
    .cb-header p {margin:0; opacity:.85; font-size:.9rem;}
    .cb-card {
        background:#fff; border:1px solid #eee; border-radius:12px;
        padding:14px 16px; margin-bottom:10px; box-shadow:0 1px 3px rgba(0,0,0,.05);
    }
    div[data-testid="stMetric"] {
        background:#faf6f2; border-radius:10px; padding:10px 6px; border:1px solid #eee;
    }
    @media (max-width: 640px){
        .cb-header h1 {font-size:1.15rem;}
        .main .block-container {padding-left:.6rem; padding-right:.6rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =======================================================================================
# 1. JSON DB ТУСЛАХ ФУНКЦУУД (UTF-8 бүрэн дэмжинэ)
# =======================================================================================
def load_json(path: str, default):
    if not os.path.exists(path):
        save_json(path, default)
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return default
            return json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_master() -> pd.DataFrame:
    data = load_json(PATH_MASTER, [])
    if not data:
        return pd.DataFrame(columns=["Код", "Нэр"])
    df = pd.DataFrame(data)
    df = df.rename(columns={"code": "Код", "name": "Нэр"})
    if "Код" not in df.columns:
        df["Код"] = ""
    if "Нэр" not in df.columns:
        df["Нэр"] = ""
    return df[["Код", "Нэр"]].astype(str)


def save_master(df: pd.DataFrame):
    records = [{"code": str(r["Код"]).strip(), "name": str(r["Нэр"]).strip()} for _, r in df.iterrows()
               if str(r["Код"]).strip() != ""]
    save_json(PATH_MASTER, records)


def empty_count_row():
    return {"Код": "", "Нэр": "", "Өглөө": 0.0, "Хүргэлт": 0.0, "Орой": 0.0, "Тайлбар": ""}


# =======================================================================================
# 2. ТУЛГАЛТЫН ЛОГИК
# =======================================================================================
def compute_actual(df: pd.DataFrame) -> pd.DataFrame:
    """Бодит = (Өглөө + Хүргэлт) - Орой"""
    df = df.copy()
    for col in ["Өглөө", "Хүргэлт", "Орой"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["Бодит"] = (df["Өглөө"] + df["Хүргэлт"]) - df["Орой"]
    return df


def find_col(columns, keywords_priority):
    """
    Багана нэрсээс түлхүүр үгтэй тохирохыг хайх (том/жижиг үсэг үл хамаарна).
    keywords_priority жагсаалтын эхний үг илүү өндөр давуу эрхтэй тул
    ойролцоо утгатай баганууд (ж: "Item #" ба "Item Name") хооронд зөв ялгана.
    """
    cols_lower = {c: str(c).lower().strip() for c in columns}
    for kw in keywords_priority:
        for c, low in cols_lower.items():
            if kw in low:
                return c
    return None


def find_header_row(raw_df: pd.DataFrame, max_scan: int = 25) -> int:
    """
    Толгой мөр нь эхний мөрөнд байхгүй тохиолдол (ж: огноо мэдээлэл дээр нь бичсэн)
    гарвал, эхний хэдэн мөрнөөс хамгийн олон түлхүүр үгтэй давхцсан мөрийг олж,
    түүнийг толгой мөр гэж тооцно.
    """
    keywords = ["item", "qty", "sold", "код", "code", "id", "нэр", "name", "plu", "price", "cost"]
    best_row, best_score = 0, -1
    for i in range(min(max_scan, len(raw_df))):
        row_vals = raw_df.iloc[i].fillna("").astype(str).str.lower().tolist()
        score = sum(1 for v in row_vals for kw in keywords if kw in v)
        if score > best_score:
            best_score, best_row = score, i
    return best_row


def clean_code(val) -> str:
    """Тоон код 465.0 маягаар унших асуудлыг засаж '465' болгоно."""
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return s


def parse_system_excel(uploaded_file) -> pd.DataFrame:
    """
    Системийн Excel-ийг уншиж (Код, Нэр, Систем) баганатай нормчилно.
    - Толгой мөр өөр газар байх (ж: эхэнд огноо мөр) тохиолдлыг автоматаар илрүүлнэ.
    - "Item #"/"Item Name" зэрэг ойролцоо нэртэй баганыг зөв ялгана.
    - Дэд нийлбэр / хоосон мөрүүдийг (Нэр хоосон байдаг) шүүж хаяна.
    - Ижил Код/Нэр давхар мөрөөр орж ирвэл (тайланд нэг бараа хэд хэдэн бүлэгт
      гарч ирдэг) тоог нь нэгтгэж нэмнэ.
    """
    uploaded_file.seek(0)
    raw = pd.read_excel(uploaded_file, engine="openpyxl", header=None)
    header_row = find_header_row(raw)

    uploaded_file.seek(0)
    df_raw = pd.read_excel(uploaded_file, engine="openpyxl", header=header_row)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    code_col = find_col(df_raw.columns, ["item #", "item#", "код", "plu", "code", "id"])
    name_col = find_col(df_raw.columns, ["item name", "нэр", "name", "бараа"])
    qty_col = find_col(df_raw.columns, ["qty sold", "qty_sold", "тоо", "sold", "qty"])

    if qty_col is None:
        raise ValueError(
            "Excel файлд 'Qty Sold' (борлуулсан тоо) багана олдсонгүй. "
            "Файлын толгой мөрийг шалгана уу."
        )
    if name_col is None and code_col is None:
        raise ValueError("Excel файлд Барааны Код эсвэл Нэр агуулсан багана олдсонгүй.")

    out = pd.DataFrame()
    out["Код"] = df_raw[code_col].apply(clean_code) if code_col else ""
    out["Нэр"] = df_raw[name_col].astype(str).str.strip() if name_col else ""
    out["Систем"] = pd.to_numeric(df_raw[qty_col], errors="coerce")

    # Дэд нийлбэр / хоосон (спэйсэр) мөрүүдийг хасах — эдгээрт Нэр хоосон байдаг
    out["Нэр"] = out["Нэр"].replace({"nan": "", "None": ""})
    out = out[out["Нэр"].str.strip() != ""]
    out = out.dropna(subset=["Систем"])

    if out.empty:
        raise ValueError(
            "Барааны мөр олдсонгүй. Excel файл дэд-нийлбэрийн мөр л агуулсан "
            "эсвэл багана буруу таарсан байж болзошгүй."
        )

    # Нэг бараа тайланд хэд хэдэн бүлэгт (цаг/ангилал зэргээр) давхардаж
    # гарч ирдэг тул Код+Нэрээр нь нэгтгэж, тоог нь нэмнэ.
    out = out.groupby(["Код", "Нэр"], as_index=False)["Систем"].sum()
    return out


def list_excel_sheets(uploaded_file):
    uploaded_file.seek(0)
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    return xls.sheet_names


def guess_default_sheet(sheet_names):
    """Тооллоготой холбоотой нэртэй хуудсыг эрхэмлэж, кассын тайлан зэргийг алгасна."""
    for s in sheet_names:
        low = s.lower()
        if "тооллого" in low or "toollogo" in low or "inventory" in low or "count" in low:
            return s
    for s in sheet_names:
        low = s.lower()
        if "касс" in low or "cash" in low or "хаалт" in low:
            continue
        return s
    return sheet_names[0]


def read_excel_with_header_guess(uploaded_file, sheet_name):
    """
    Толгой мөр эхний мөрөнд биш байх тохиолдлыг (ж: дээр нь огноо/гарчиг мөр байх)
    автоматаар илрүүлж, тухайн мөрөөр header болгож уншина.
    """
    uploaded_file.seek(0)
    raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl", header=None)
    keywords = ["код", "№", "id", "plu", "нэр", "name", "өглөө", "morning", "хүргэлт", "орлого",
                "delivery", "орой", "evening", "гаралт", "систем", "зөрүү", "тайлбар", "comment", "note"]
    best_row, best_score = 0, -1
    for i in range(min(40, len(raw))):
        row_vals = raw.iloc[i].fillna("").astype(str).str.lower().tolist()
        score = sum(1 for v in row_vals for kw in keywords if kw in v)
        if score > best_score:
            best_score, best_row = score, i
    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl", header=best_row)
    df.columns = [str(c).strip() for c in df.columns]
    # Бүрэн хоосон мөр/баганыг цэвэрлэх
    df = df.dropna(axis=0, how="all")
    return df, best_row


def guess_count_column_defaults(df: pd.DataFrame) -> dict:
    """Тооллогын баганууд (Код/Нэр/Өглөө/Хүргэлт/Орой/Тайлбар)-ыг эхлэн таамаглана."""
    cols = list(df.columns)
    g = {
        "code": find_col(cols, ["код", "№", "no", "plu", "id"]),
        "name": find_col(cols, ["нэр", "name", "бараа"]),
        "morning": find_col(cols, ["өглөө", "morning"]),
        "delivery": find_col(cols, ["хүргэлт", "орлого", "delivery"]),
        "evening": find_col(cols, ["орой", "evening"]),
        "note": find_col(cols, ["тайлбар", "comment", "note"]),
    }
    # Зарим загварт "Өглөө" баганад нэр байдаггүй ("Хүргэлт" баганын
    # шууд зүүн талд байдаг нийтлэг хэлбэр) — тухайн баганад тоон утга
    # байгаа эсэхийг шалгаад таамаглал болгоно.
    if g["morning"] is None and g["delivery"] is not None:
        idx = cols.index(g["delivery"])
        if idx > 0:
            candidate = cols[idx - 1]
            if pd.to_numeric(df[candidate], errors="coerce").notna().sum() > 0:
                g["morning"] = candidate
    return g


# =======================================================================================
# OCR — Тооллогын хүснэгтийн ЗУРАГ (screenshot)-аас тоо унших
# =======================================================================================
# Арга барил: (1) бүх зургийг нэг удаа OCR хийж, торны шугамын (vertical grid line)
# байрлалаас баганы хил (Код/Нэр/Өглөө/Хүргэлт/Орой/... ) болон текстийн мөрүүдийн
# Y-координатын төвийг тодорхойлно; (2) дараа нь нүд (мөр × багана) БҮРИЙГ ТУСАД НЬ
# зурагнаас таслаж, тоон баганад зөвхөн тоо унших горим (digit whitelist)-оор дахин
# OCR хийнэ. Ингэснээр нэг мөр OCR-д алдагдвал дараагийн бүх мөр шилждэг эрсдэлгүй
# (нүд бүр өөрийн байрлалдаа шууд холбогддог тул).

_OCR_HEADER_KEYWORDS = ["код", "№", "нэр", "өглөө", "morning", "хүргэлт", "орлого",
                        "delivery", "орой", "evening", "гаралт", "систем", "зөрүү",
                        "тайлбар", "comment", "note"]


def _ocr_words_and_pipes(img, lang="eng+mon"):
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DATAFRAME, config="--psm 6")
    data = data.dropna(subset=["text"])
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    pipe_mask = data["text"].str.match(r'^[\|\}\{\[\]_~`]+$')
    pipes = data[pipe_mask]
    words = data[~pipe_mask].copy()
    # Зурагны хамгийн зүүн ирмэг дэх жижиг artifact (мөрийн дугаарын багана) хасах
    words = words[~((words["left"] < 50) & (words["width"] < 40))]
    words = words[words["conf"] > 10]
    return words, pipes, data["line_num"].nunique()


def _ocr_column_boundaries(pipes: pd.DataFrame, n_lines: int):
    if len(pipes) < 5:
        return []
    lefts_sorted = np.sort(pipes["left"].values)
    clusters, cur = [], [lefts_sorted[0]]
    for x in lefts_sorted[1:]:
        if x - cur[-1] < 60:
            cur.append(x)
        else:
            clusters.append(cur)
            cur = [x]
    clusters.append(cur)
    min_count = max(3, 0.3 * n_lines)
    boundaries = sorted(float(np.mean(c)) for c in clusters if len(c) >= min_count)
    return [b for b in boundaries if b > 60]


def _ocr_row_centers(words: pd.DataFrame):
    w = words.copy()
    w["line_key"] = w["block_num"].astype(str) + "_" + w["par_num"].astype(str) + "_" + w["line_num"].astype(str)
    centers = []
    for _, grp in w.groupby("line_key"):
        centers.append(float(grp["top"].mean() + grp["height"].mean() / 2))
    return sorted(centers)


def _ocr_bucket_of(x, boundaries):
    for i, b in enumerate(boundaries):
        if x < b:
            return i
    return len(boundaries)


def _ocr_find_header(words: pd.DataFrame, row_centers, col_boundaries):
    """Толгой мөрийг олж, багана бүрийг (Код/Нэр/Өглөө/Хүргэлт/Орой/Тайлбар) тодорхойлно."""
    w = words.copy()
    w["line_key"] = w["block_num"].astype(str) + "_" + w["par_num"].astype(str) + "_" + w["line_num"].astype(str)
    line_groups = {key: grp for key, grp in w.groupby("line_key")}

    n_buckets = len(col_boundaries) + 1
    best_idx, best_score, best_field_map = None, -1, None
    for i, y in enumerate(row_centers):
        # тухайн Y-тэй ойролцоо (±15px) line_key-г олох
        match_grp = None
        for key, grp in line_groups.items():
            gy = grp["top"].mean() + grp["height"].mean() / 2
            if abs(gy - y) < 15:
                match_grp = grp
                break
        if match_grp is None:
            continue
        bucket_text = {b: [] for b in range(n_buckets)}
        for _, r in match_grp.iterrows():
            b = _ocr_bucket_of(r["left"] + r["width"] / 2, col_boundaries)
            bucket_text[b].append(r["text"].lower())
        field_map, score = {}, 0
        for b, wl in bucket_text.items():
            joined = " ".join(wl)
            field = None
            if "нэр" in joined or "name" in joined:
                field = "Нэр"; score += 1
            elif "өглөө" in joined or "morning" in joined:
                field = "Өглөө"; score += 1
            elif "хүргэлт" in joined or "орлого" in joined:
                field = "Хүргэлт"; score += 1
            elif "орой" in joined or "evening" in joined:
                field = "Орой"; score += 1
            elif "тайлбар" in joined or "comment" in joined:
                field = "Тайлбар"; score += 1
            elif "код" in joined or "№" in joined:
                field = "Код"; score += 1
            field_map[b] = field
        if score > best_score:
            best_score, best_idx, best_field_map = score, i, field_map

    if best_idx is None or best_score < 2:
        return None, None
    return best_idx, best_field_map


def ocr_extract_count_table(image_bytes, lang="eng+mon", upscale=3, progress_cb=None):
    """
    Тооллогын хүснэгтийн screenshot-оос Код/Нэр/Өглөө/Хүргэлт/Орой/Тайлбар баганыг
    уншиж, pd.DataFrame буцаана. Найдваргүй бол ValueError өргөнө.
    """
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    if upscale and upscale != 1:
        img = img.resize((img.width * upscale, img.height * upscale), PILImage.LANCZOS)
    W, H = img.size

    words, pipes, n_lines = _ocr_words_and_pipes(img, lang=lang)
    if len(words) < 5:
        raise ValueError("Зургаас текст олдсонгүй. Илүү тод/өндөр нягтралтай зураг оруулна уу.")

    col_boundaries = _ocr_column_boundaries(pipes, n_lines)
    row_centers = _ocr_row_centers(words)
    if len(col_boundaries) < 2:
        raise ValueError(
            "Баганын хилийг тодорхойлж чадсангүй. Энэ горим нь торны шугам (cell border) "
            "тодорхой харагдах Excel screenshot дээр хамгийн сайн ажилладаг."
        )

    header_idx, field_map = _ocr_find_header(words, row_centers, col_boundaries)
    if header_idx is None:
        raise ValueError(
            "Толгой мөр (Код/Нэр/Өглөө/Хүргэлт/Орой гэсэн бичээстэй мөр) олдсонгүй. "
            "Зурагт толгой мөр бүрэн, тод харагдаж байгаа эсэхийг шалгана уу."
        )

    col_edges = [0] + col_boundaries + [W]
    numeric_fields = {"Өглөө", "Хүргэлт", "Орой"}
    text_fields = {"Нэр", "Тайлбар", "Код"}

    data_row_ys = row_centers[header_idx + 1:]
    row_bounds = [0.0]
    for i in range(len(data_row_ys) - 1):
        row_bounds.append((data_row_ys[i] + data_row_ys[i + 1]) / 2)
    row_bounds.append(float(H))

    results = []
    total = max(1, len(data_row_ys))
    for r_i in range(len(data_row_ys)):
        y0, y1 = row_bounds[r_i], row_bounds[r_i + 1]
        row_out = {"Код": "", "Нэр": "", "Өглөө": "", "Хүргэлт": "", "Орой": "", "Тайлбар": ""}
        for b, field in field_map.items():
            if field is None:
                continue
            x0, x1 = col_edges[b], col_edges[b + 1]
            pad = 4
            crop = img.crop((max(0, int(x0) + pad), int(y0) + pad, int(x1) - pad, max(int(y0) + pad + 1, int(y1) - pad)))
            if field in numeric_fields:
                cfg = "--psm 7 -c tessedit_char_whitelist=0123456789.,-"
                txt = pytesseract.image_to_string(crop, lang="eng", config=cfg)
            else:
                txt = pytesseract.image_to_string(crop, lang=lang, config="--psm 7")
            row_out[field] = re.sub(r"\s+", " ", txt).strip()
        results.append(row_out)
        if progress_cb:
            progress_cb((r_i + 1) / total)

    df = pd.DataFrame(results)

    def _clean_num(s):
        if not s:
            return 0.0
        m = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
        if not m:
            return 0.0
        try:
            return float(m[-1])
        except ValueError:
            return 0.0

    for col in ["Өглөө", "Хүргэлт", "Орой"]:
        if col in df.columns:
            df[col] = df[col].apply(_clean_num)
        else:
            df[col] = 0.0
    for col in ["Код", "Нэр", "Тайлбар"]:
        if col not in df.columns:
            df[col] = ""

    # Тайлбар баганад ихэвчлэн торны шугамын chimээ (зөвхөн тэмдэгт/тоо, үсэггүй)
    # орох тул зөвхөн үсэг агуулсан утгыг хадгална
    letters_re = re.compile(r"[A-Za-zА-Яа-яЁёӨөҮү]")
    df["Тайлбар"] = df["Тайлбар"].apply(lambda s: s if letters_re.search(s or "") else "")

    df = df[df["Нэр"].str.strip() != ""]
    df = df[~df["Нэр"].str.contains("mpos|cpos", case=False, na=False)]
    df = df.reset_index(drop=True)
    return df[["Код", "Нэр", "Өглөө", "Хүргэлт", "Орой", "Тайлбар"]]


def reconcile(df_count: pd.DataFrame, df_system: pd.DataFrame) -> pd.DataFrame:
    """
    Код-оор эхлээд тулгана, олдохгүй бол Fuzzy search-ээр нэрээр тулгана.
    Зөрүү = Бодит - Систем
    """
    df = compute_actual(df_count)

    code_map = {}
    if "Код" in df_system.columns:
        for _, r in df_system.iterrows():
            code = str(r["Код"]).strip()
            if code and code.lower() != "nan":
                code_map[code] = r["Систем"]

    name_map = {}
    if "Нэр" in df_system.columns:
        for _, r in df_system.iterrows():
            nm = str(r["Нэр"]).strip()
            if nm and nm.lower() != "nan":
                name_map[nm] = r["Систем"]
    system_names = list(name_map.keys())

    system_qty_list = []
    match_method_list = []

    for _, row in df.iterrows():
        code = str(row.get("Код", "")).strip()
        name = str(row.get("Нэр", "")).strip()
        sys_qty = None
        method = "Олдсонгүй"

        # 1) Код-оор тулгах ("0" гэдгийг "код онооогүй" гэж үзээд алгасна —
        #    учир нь олон бараа код=0 хуваалцдаг тул анхаарамжгүй таарч болзошгүй)
        if code and code != "0" and code in code_map:
            sys_qty = code_map[code]
            method = "Код"
        # 2) Fuzzy search — нэрээр тулгах
        elif name and system_names:
            best = process.extractOne(name, system_names, scorer=fuzz.token_sort_ratio)
            if best and best[1] >= FUZZY_THRESHOLD:
                sys_qty = name_map[best[0]]
                method = f"Fuzzy ({best[1]}%) → {best[0]}"

        if sys_qty is None:
            sys_qty = 0.0

        system_qty_list.append(sys_qty)
        match_method_list.append(method)

    df["Систем"] = system_qty_list
    df["Тулгасан аргаас"] = match_method_list
    df["Зөрүү"] = df["Бодит"] - df["Систем"]
    return df


def color_diff(val):
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v < 0:
        return "color:#c0392b; font-weight:700; background-color:#fdecea;"
    elif v > 0:
        return "color:#1e8449; font-weight:700; background-color:#eafaf1;"
    return "color:#555;"


def styled_table(df: pd.DataFrame, diff_col="Зөрүү"):
    cols = [c for c in df.columns if c in
            ["Код", "Нэр", "Өглөө", "Хүргэлт", "Орой", "Бодит", "Систем", "Зөрүү", "Тулгасан аргаас", "Тайлбар"]]
    view = df[cols] if cols else df
    sty = view.style
    if diff_col in view.columns:
        # pandas 2.1-с "applymap" нэрийг "map" болгож сольсон бөгөөд хамгийн
        # шинэ хувилбаруудад "applymap" бүрмөсөн устсан тул хоёуланг нь дэмжинэ.
        if hasattr(sty, "map"):
            sty = sty.map(color_diff, subset=[diff_col])
        else:
            sty = sty.applymap(color_diff, subset=[diff_col])
    fmt = {c: "{:.1f}" for c in ["Өглөө", "Хүргэлт", "Орой", "Бодит", "Систем", "Зөрүү"] if c in view.columns}
    sty = sty.format(fmt)
    return sty


def df_to_excel_bytes(df: pd.DataFrame, sheet_name="Тайлан") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


# =======================================================================================
# БАГЦ (сарын) импорт — олон өдрийн Excel файлыг огноогоор автоматаар тааруулж тулгах
# =======================================================================================
_DATE_RE_DOTTED = re.compile(r'(20\d{2})[.\-/_](\d{1,2})[.\-/_](\d{1,2})')
_DATE_RE_US = re.compile(r'(\d{1,2})/(\d{1,2})/(20\d{2})')


def _safe_date(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def extract_date_from_filename(filename: str):
    m = _DATE_RE_DOTTED.search(filename)
    if m:
        return _safe_date(*m.groups())
    return None


def extract_date_from_toollogo(uploaded_file, sheet_name: str):
    """toollogo хуудасны аль нэг нүднээс YYYY.M.D / YYYY-M-D маягийн огноог хайна
    (ихэвчлэн 'mpos ... cpos ... 2026.9.2' гэсэн footer мөрөнд байдаг)."""
    try:
        uploaded_file.seek(0)
        raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl", header=None)
    except Exception:
        return None
    # Сүүлээс нь эхлэн хайх (footer-т байх магадлал өндөр)
    for i in range(len(raw) - 1, -1, -1):
        row_text = " ".join(raw.iloc[i].fillna("").astype(str).tolist())
        m = _DATE_RE_DOTTED.search(row_text)
        if m:
            d = _safe_date(*m.groups())
            if d:
                return d
    return None


def extract_date_from_system_file(uploaded_file, sheet_name: str):
    """Системийн Excel-ийн 'Date: 9/2/2026 12:00:00 AM to ...' мөрөөс огноо ялгана."""
    try:
        uploaded_file.seek(0)
        raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl", header=None)
    except Exception:
        return None
    for i in range(min(15, len(raw))):
        row_text = " ".join(raw.iloc[i].fillna("").astype(str).tolist())
        m = _DATE_RE_US.search(row_text)
        if m:
            mo, d, y = m.groups()
            dt = _safe_date(y, mo, d)
            if dt:
                return dt
    return None


def batch_reconcile_files(toollogo_files, system_files):
    """
    Олон тооллогын Excel болон олон системийн Excel файлыг огноогоор нь тааруулж,
    тус бүрийг тулгаад жагсаалт болгож буцаана:
    [{"date", "status", "n_items", "reconciled_df", "error"}, ...]
    """
    tool_by_date = {}
    for f in toollogo_files:
        try:
            sheets = list_excel_sheets(f)
            sheet = guess_default_sheet(sheets)
            d = extract_date_from_toollogo(f, sheet) or extract_date_from_filename(f.name)
            key = d if d else f"❓ {f.name}"
            tool_by_date[key] = (f, sheet)
        except Exception:
            tool_by_date[f"❓ {f.name}"] = (f, None)

    sys_by_date = {}
    for f in system_files:
        try:
            sheets = list_excel_sheets(f)
            d = extract_date_from_system_file(f, sheets[0]) or extract_date_from_filename(f.name)
            key = d if d else f"❓ {f.name}"
            sys_by_date[key] = (f, sheets[0])
        except Exception:
            sys_by_date[f"❓ {f.name}"] = (f, None)

    all_keys = sorted(set(tool_by_date.keys()) | set(sys_by_date.keys()), key=lambda x: str(x))
    results = []
    for key in all_keys:
        entry = {"date": key, "status": "", "n_items": 0, "reconciled_df": None, "error": None}
        tool_entry = tool_by_date.get(key)
        sys_entry = sys_by_date.get(key)

        if tool_entry is None:
            entry["status"] = "⚠️ Тооллогын файл дутуу (зөвхөн систем)"
            results.append(entry)
            continue
        if sys_entry is None:
            entry["status"] = "⚠️ Системийн файл дутуу (зөвхөн тооллого)"
            results.append(entry)
            continue

        try:
            tf, tsheet = tool_entry
            sf, _ = sys_entry
            if tsheet is None:
                raise ValueError("Тооллогын хуудсыг тодорхойлж чадсангүй")

            df_import, _ = read_excel_with_header_guess(tf, tsheet)
            guesses = guess_count_column_defaults(df_import)
            if not guesses["code"] or not guesses["name"]:
                raise ValueError("Код/Нэр баганыг автоматаар олж чадсангүй")

            count_df = pd.DataFrame()
            count_df["Код"] = df_import[guesses["code"]].apply(clean_code)
            count_df["Нэр"] = df_import[guesses["name"]].astype(str).str.strip().replace(
                {"nan": "", "None": ""})
            count_df["Өглөө"] = (pd.to_numeric(df_import[guesses["morning"]], errors="coerce").fillna(0.0)
                                  if guesses["morning"] else 0.0)
            count_df["Хүргэлт"] = (pd.to_numeric(df_import[guesses["delivery"]], errors="coerce").fillna(0.0)
                                    if guesses["delivery"] else 0.0)
            count_df["Орой"] = (pd.to_numeric(df_import[guesses["evening"]], errors="coerce").fillna(0.0)
                                 if guesses["evening"] else 0.0)
            count_df["Тайлбар"] = (df_import[guesses["note"]].astype(str).replace({"nan": "", "None": ""})
                                    if guesses["note"] else "")
            count_df = count_df[count_df["Нэр"].str.strip() != ""]
            count_df = count_df[~count_df["Нэр"].str.contains("mpos|cpos", case=False, na=False)]
            count_df = count_df.reset_index(drop=True)

            sys_df = parse_system_excel(sf)
            reconciled = reconcile(count_df, sys_df)

            entry["reconciled_df"] = reconciled
            entry["n_items"] = len(reconciled)
            entry["status"] = "✅ Бэлэн"
        except Exception as e:
            entry["status"] = f"❌ Алдаа: {e}"
            entry["error"] = str(e)
        results.append(entry)
    return results


# =======================================================================================
# 3. SESSION STATE ЭХЛҮҮЛЭХ
# =======================================================================================
if "count_df" not in st.session_state:
    saved_current = load_json(PATH_CURRENT, None)
    if saved_current and saved_current.get("items"):
        st.session_state.count_df = pd.DataFrame(saved_current["items"])
    else:
        st.session_state.count_df = pd.DataFrame([empty_count_row()])

if "reconciled_df" not in st.session_state:
    st.session_state.reconciled_df = None


# =======================================================================================
# 4. HEADER
# =======================================================================================
st.markdown(
    """
    <div class="cb-header">
        <h1>☕ Caffe Bene — Тооллого & Борлуулалтын систем тулгалт</h1>
        <p>Өдрийн тооллого · Системтэй тулгах · Архив & Хогийн сав · Барааны мэдээллийн сан</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["📝 ТООЛЛОГО", "📊 АРХИВ"])


# =======================================================================================
# TAB 1 — ТООЛЛОГО
# =======================================================================================
with tab1:
    st.subheader("📝 Өдрийн тооллого")

    col_date, col_btn1, col_btn2 = st.columns([2, 1, 1])
    with col_date:
        count_date = st.date_input("Тооллогын огноо", value=date.today())

    # -----------------------------------------------------------------------------
    # Excel файлаас Ø/Х/О тоог ачаалах (гараар шивэхийн оронд)
    # -----------------------------------------------------------------------------
    with st.expander("📥 Тооллогыг Excel файлаас ачаалах (гараар шивэхийн оронд)", expanded=False):
        st.caption(
            "Өөрийн ажлын Excel файлаа (Код/Нэр/Өглөө/Хүргэлт/Орой/Тайлбар баганатай) upload "
            "хийгээд баганыг доор тохируулаад ачаална. Толгой мөр, баганын байршил ямар ч "
            "байсан автоматаар таамаглаж, шаардлагатай бол гараар засах боломжтой."
        )
        count_file = st.file_uploader("Тооллогын Excel файл", type=["xlsx", "xls"], key="count_upload")

        if count_file is not None:
            try:
                sheet_names = list_excel_sheets(count_file)
                if len(sheet_names) > 1:
                    default_sheet = guess_default_sheet(sheet_names)
                    sel_sheet = st.selectbox(
                        "Хуудас (Sheet) сонгох", sheet_names,
                        index=sheet_names.index(default_sheet),
                    )
                else:
                    sel_sheet = sheet_names[0]

                df_import, header_row_idx = read_excel_with_header_guess(count_file, sel_sheet)
                st.caption(f"'{sel_sheet}' хуудасны {header_row_idx + 1}-р мөрийг толгой мөр гэж "
                           f"тооцож уншлаа. Эхний 5 мөр:")
                st.dataframe(df_import.head(5), use_container_width=True, hide_index=True)

                guesses = guess_count_column_defaults(df_import)
                cols_all = list(df_import.columns)
                opts_req = cols_all
                opts_opt = ["— Байхгүй (0 / хоосон) —"] + cols_all

                def _idx(opts, val):
                    try:
                        return opts.index(val)
                    except (ValueError, TypeError):
                        return 0

                st.write("**Баганын харгалзаа:**")
                c1, c2, c3 = st.columns(3)
                code_sel = c1.selectbox("Код багана", opts_req, index=_idx(opts_req, guesses["code"]))
                name_sel = c2.selectbox("Нэр багана", opts_req, index=_idx(opts_req, guesses["name"]))
                morning_sel = c3.selectbox("Өглөө багана", opts_opt, index=_idx(opts_opt, guesses["morning"]))

                c4, c5, c6 = st.columns(3)
                delivery_sel = c4.selectbox("Хүргэлт багана", opts_opt, index=_idx(opts_opt, guesses["delivery"]))
                evening_sel = c5.selectbox("Орой багана", opts_opt, index=_idx(opts_opt, guesses["evening"]))
                note_sel = c6.selectbox("Тайлбар багана", opts_opt, index=_idx(opts_opt, guesses["note"]))

                load_mode = st.radio(
                    "Ачаалах горим",
                    ["Одоогийн хүснэгтийг орлуулах", "Одоогийн хүснэгтэд нэмж холбох"],
                    horizontal=True,
                )

                if st.button("📥 Тооллогын хүснэгтэд ачаалах", type="primary", use_container_width=True):
                    def _num_col(sel):
                        if sel == "— Байхгүй (0 / хоосон) —":
                            return pd.Series(0.0, index=df_import.index)
                        return pd.to_numeric(df_import[sel], errors="coerce").fillna(0.0)

                    def _text_col(sel):
                        if sel == "— Байхгүй (0 / хоосон) —":
                            return pd.Series("", index=df_import.index)
                        return (df_import[sel].astype(str)
                                .replace({"nan": "", "None": ""}).str.strip())

                    new_df = pd.DataFrame()
                    new_df["Код"] = df_import[code_sel].apply(clean_code)
                    new_df["Нэр"] = df_import[name_sel].astype(str).str.strip().replace(
                        {"nan": "", "None": ""})
                    new_df["Өглөө"] = _num_col(morning_sel)
                    new_df["Хүргэлт"] = _num_col(delivery_sel)
                    new_df["Орой"] = _num_col(evening_sel)
                    new_df["Тайлбар"] = _text_col(note_sel)

                    # Хоосон нэртэй мөрүүдийг (ж: дэд-нийлбэр, хоосон зай мөр) хасах
                    new_df = new_df[new_df["Нэр"].str.strip() != ""]
                    new_df = new_df.reset_index(drop=True)

                    if load_mode == "Одоогийн хүснэгтийг орлуулах":
                        st.session_state.count_df = new_df
                    else:
                        st.session_state.count_df = pd.concat(
                            [st.session_state.count_df, new_df], ignore_index=True
                        )
                    st.success(f"{len(new_df)} мөр амжилттай ачааллаа. Доорх хүснэгтээс шалгана уу.")
                    st.rerun()
            except Exception as e:
                st.error(f"Файл уншихад алдаа гарлаа: {e}")

    # -----------------------------------------------------------------------------
    # Зургаас (screenshot) уншуулах — OCR
    # -----------------------------------------------------------------------------
    with st.expander("🖼️ Тооллогын зургаас (screenshot) уншуулах — OCR", expanded=False):
        if not OCR_LIBS_AVAILABLE:
            st.warning(
                "OCR сан (pytesseract / Pillow) энэ орчинд суугаагүй тул энэ боломж идэвхгүй "
                "байна. `requirements.txt`-д `pytesseract`, `Pillow`, `packages.txt`-д "
                "`tesseract-ocr`, `tesseract-ocr-mon` нэмээд дахин deploy хийнэ үү."
            )
        else:
            st.warning(
                "⚠️ **Зургаас тоо таних (OCR) 100% үнэн зөв биш.** Уншсаны дараа гарч ирэх "
                "хүснэгтийг эх зурагтайгаа **заавал тулгаж шалгаад**, алдаатай тоог засаад "
                "ачаална уу — ялангуяа тоон утгууд дээр анхаарна уу."
            )
            st.caption(
                "Хамгийн сайн ажиллах нөхцөл: торон шугам (cell border) тод харагдах "
                "Excel screenshot. Гар бичмэл эсвэл өнцгөөр гажсан зураг дээр нарийвчлал буурна."
            )
            ocr_image = st.file_uploader(
                "Тооллогын хүснэгтийн зураг", type=["png", "jpg", "jpeg"], key="ocr_upload"
            )

            if ocr_image is not None and st.button("🔍 Зургаас унших", key="ocr_run_btn"):
                progress_bar = st.progress(0.0, text="Зургаас өгөгдөл уншиж байна...")
                try:
                    img_bytes = ocr_image.getvalue()
                    ocr_df = ocr_extract_count_table(
                        img_bytes,
                        progress_cb=lambda p: progress_bar.progress(
                            min(p, 1.0), text=f"Уншиж байна... {int(min(p,1.0)*100)}%"
                        ),
                    )
                    progress_bar.empty()
                    st.session_state.ocr_preview_df = ocr_df
                    if len(ocr_df) == 0:
                        st.warning("Барааны мөр олдсонгүй. Зургаа шалгаад дахин оруулна уу.")
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Уншихад алдаа гарлаа: {e}")

            preview = st.session_state.get("ocr_preview_df")
            if preview is not None and len(preview) > 0:
                st.write(f"**Уншсан үр дүн ({len(preview)} мөр) — ЗААВАЛ ШАЛГАЖ ЗАСААРАЙ:**")
                ocr_edited = st.data_editor(
                    preview,
                    use_container_width=True, hide_index=True, num_rows="dynamic",
                    key="ocr_preview_editor",
                    column_config={
                        "Код": st.column_config.TextColumn("Код", width="small"),
                        "Нэр": st.column_config.TextColumn("Нэр", width="medium"),
                        "Өглөө": st.column_config.NumberColumn("Өглөө", format="%.1f"),
                        "Хүргэлт": st.column_config.NumberColumn("Хүргэлт", format="%.1f"),
                        "Орой": st.column_config.NumberColumn("Орой", format="%.1f"),
                        "Тайлбар": st.column_config.TextColumn("Тайлбар", width="medium"),
                    },
                )
                ocr_load_mode = st.radio(
                    "Ачаалах горим",
                    ["Одоогийн хүснэгтийг орлуулах", "Одоогийн хүснэгтэд нэмж холбох"],
                    horizontal=True, key="ocr_load_mode",
                )
                if st.button("✅ Шалгасан өгөгдлийг тооллогын хүснэгтэд ачаалах",
                              type="primary", use_container_width=True, key="ocr_confirm_btn"):
                    if ocr_load_mode == "Одоогийн хүснэгтийг орлуулах":
                        st.session_state.count_df = ocr_edited.reset_index(drop=True)
                    else:
                        st.session_state.count_df = pd.concat(
                            [st.session_state.count_df, ocr_edited], ignore_index=True
                        ).reset_index(drop=True)
                    st.session_state.ocr_preview_df = None
                    st.success("Тооллогын хүснэгтэд ачааллаа.")
                    st.rerun()

    st.caption("Мөр бүрт Өглөө / Хүргэлт (Орлого) / Орой-ийн тоог оруулна уу. "
               "Шинэ мөр нэмэхдээ хүснэгтийн доод хэсгийн **+** товч ашиглана.")

    edited_df = st.data_editor(
        st.session_state.count_df,
        num_rows="dynamic",
        use_container_width=True,
        key="count_editor",
        column_config={
            "Код": st.column_config.TextColumn("Код (PLU)", width="small"),
            "Нэр": st.column_config.TextColumn("Барааны нэр", width="medium"),
            "Өглөө": st.column_config.NumberColumn("Өглөө (Ө)", min_value=0.0, step=1.0, format="%.1f"),
            "Хүргэлт": st.column_config.NumberColumn("Хүргэлт/Орлого (Х)", min_value=0.0, step=1.0, format="%.1f"),
            "Орой": st.column_config.NumberColumn("Орой (О)", min_value=0.0, step=1.0, format="%.1f"),
            "Тайлбар": st.column_config.TextColumn("Тайлбар", width="large"),
        },
        hide_index=True,
    )

    st.session_state.count_df = edited_df

    calc_df = compute_actual(edited_df)
    total_actual = calc_df["Бодит"].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Мөрийн тоо", len(calc_df))
    m2.metric("Нийт Бодит (Ө+Х-О)", f"{total_actual:,.1f}")
    m3.metric("Огноо", count_date.strftime("%Y-%m-%d"))

    st.write("**Тооцоолсон Бодит зарагдсан тоо (тулгалтын өмнөх):**")
    st.dataframe(
        calc_df[["Код", "Нэр", "Өглөө", "Хүргэлт", "Орой", "Бодит", "Тайлбар"]],
        use_container_width=True, hide_index=True,
    )

    col_save1, col_save2 = st.columns(2)
    with col_save1:
        if st.button("💾 Түр хадгалах (Draft)", use_container_width=True):
            save_json(PATH_CURRENT, {
                "date": count_date.strftime("%Y-%m-%d"),
                "items": edited_df.to_dict(orient="records"),
                "saved_at": datetime.now().isoformat(),
            })
            st.success("Түр хадгаллаа. Дараа нэвтрэхэд энэ өгөгдөл сэргэнэ.")
    with col_save2:
        if st.button("🗑️ Хүснэгтийг цэвэрлэх", use_container_width=True):
            st.session_state.count_df = pd.DataFrame([empty_count_row()])
            st.session_state.reconciled_df = None
            st.rerun()

    st.divider()
    st.subheader("🔄 Системийн Excel-тэй тулгах")
    st.caption("Excel файл нь `Код`/`ID`, `Нэр`(заавал биш) болон **`Qty Sold`** баганатай байх ёстой.")

    sys_file = st.file_uploader("Системийн борлуулалтын Excel файл", type=["xlsx", "xls"], key="sys_upload")

    if sys_file is not None:
        try:
            df_system = parse_system_excel(sys_file)
            st.success(f"Системийн файлаас {len(df_system)} мөр амжилттай уншлаа.")
            if st.button("⚖️ Тулгалт хийх", type="primary", use_container_width=True):
                reconciled = reconcile(edited_df, df_system)
                st.session_state.reconciled_df = reconciled
        except Exception as e:
            st.error(f"Файл уншихад алдаа гарлаа: {e}")

    if st.session_state.reconciled_df is not None:
        rdf = st.session_state.reconciled_df
        st.write("**Тулгалтын үр дүн** (🟥 Дутсан — Улаан | 🟩 Илүүдсэн — Ногоон):")
        st.dataframe(styled_table(rdf), use_container_width=True, hide_index=True)

        d1, d2, d3 = st.columns(3)
        d1.metric("Дутсан барааны тоо", int((rdf["Зөрүү"] < 0).sum()))
        d2.metric("Илүүдсэн барааны тоо", int((rdf["Зөрүү"] > 0).sum()))
        d3.metric("Тохирсон барааны тоо", int((rdf["Зөрүү"] == 0).sum()))

        st.write("**📷 Нотлох баримт хавсаргах (заавал биш)**")
        st.caption("Кассын хуудас, гар бичмэл тооллого гэх мэт зургуудыг энд хавсаргавал "
                   "архивтай хамт хадгалагдана. Хэд хэдэн зураг зэрэг сонгож болно.")
        evidence_photos = st.file_uploader(
            "Зураг хавсаргах", type=["png", "jpg", "jpeg"],
            accept_multiple_files=True, key="evidence_photo_upload",
        )

        if st.button("📦 Архивлах (Тулгалтыг баталгаажуулж хадгалах)", type="primary", use_container_width=True):
            record_id = str(uuid.uuid4())

            # Хавсаргасан зургуудыг диск дээр хадгалах
            photo_paths = []
            if evidence_photos:
                record_photo_dir = os.path.join(PATH_PHOTOS, record_id)
                os.makedirs(record_photo_dir, exist_ok=True)
                for i, photo in enumerate(evidence_photos):
                    ext = os.path.splitext(photo.name)[1] or ".jpg"
                    fname = f"{i+1:02d}{ext}"
                    fpath = os.path.join(record_photo_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(photo.getbuffer())
                    photo_paths.append(fpath)

            history = load_json(PATH_HISTORY, [])
            record = {
                "id": record_id,
                "date": count_date.strftime("%Y-%m-%d"),
                "archived_at": datetime.now().isoformat(),
                "items": rdf.to_dict(orient="records"),
                "photos": photo_paths,
            }
            history.append(record)
            save_json(PATH_HISTORY, history)

            # Түр хадгалалтыг цэвэрлэх
            save_json(PATH_CURRENT, {"date": "", "items": [], "saved_at": ""})
            st.session_state.count_df = pd.DataFrame([empty_count_row()])
            st.session_state.reconciled_df = None
            st.success(f"{count_date.strftime('%Y-%m-%d')} өдрийн тооллого архивлагдлаа!"
                       + (f" ({len(photo_paths)} зурагтай)" if photo_paths else ""))
            st.rerun()


# =======================================================================================
# TAB 2 — АРХИВ
# =======================================================================================
with tab2:
    st.subheader("📊 Тулгалтын түүх / Архив")

    with st.expander("📥 Сарын файлуудыг багцаар оруулж тулгах", expanded=False):
        st.caption(
            "Хэд хэдэн өдрийн **тооллогын Excel** файл болон **системийн Excel** файлыг "
            "зэрэг оруулбал, файл дотроо байгаа огноогоор нь (эсвэл файлын нэрэнд "
            "2026_9_2 гэх мэт огноо байвал) автоматаар тааруулж тулгаад, шалгасны дараа "
            "бүгдийг АРХИВ-т нэг дор хадгална."
        )
        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            batch_tool_files = st.file_uploader(
                "Тооллогын Excel файлууд", type=["xlsx", "xls"],
                accept_multiple_files=True, key="batch_tool_files",
            )
        with col_bt2:
            batch_sys_files = st.file_uploader(
                "Системийн Excel файлууд", type=["xlsx", "xls"],
                accept_multiple_files=True, key="batch_sys_files",
            )

        if batch_tool_files and batch_sys_files:
            if st.button("🔎 Огноогоор тааруулж тулгах", key="batch_match_btn", use_container_width=True):
                with st.spinner("Файлуудыг уншиж, огноогоор тааруулж тулгаж байна..."):
                    st.session_state.batch_results = batch_reconcile_files(batch_tool_files, batch_sys_files)

        batch_results = st.session_state.get("batch_results")
        if batch_results:
            st.write("**Тулгалтын явц:**")
            summary_rows = [{"Огноо": str(r["date"]), "Статус": r["status"], "Мөр": r["n_items"]}
                             for r in batch_results]
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            ready = [r for r in batch_results if r["reconciled_df"] is not None]
            if ready:
                preview_labels = [str(r["date"]) for r in ready]
                sel_batch_date = st.selectbox("👁️ Дэлгэрэнгүй харах өдөр", preview_labels, key="batch_preview_sel")
                sel_r = next(r for r in ready if str(r["date"]) == sel_batch_date)
                st.dataframe(styled_table(sel_r["reconciled_df"]), use_container_width=True, hide_index=True)

                if st.button(f"📦 Бэлэн болсон {len(ready)} өдрийг бүгдийг архивлах",
                              type="primary", use_container_width=True, key="batch_archive_btn"):
                    hist = load_json(PATH_HISTORY, [])
                    for r in ready:
                        hist.append({
                            "id": str(uuid.uuid4()),
                            "date": str(r["date"]),
                            "archived_at": datetime.now().isoformat(),
                            "items": r["reconciled_df"].to_dict(orient="records"),
                            "photos": [],
                        })
                    save_json(PATH_HISTORY, hist)
                    st.session_state.batch_results = None
                    st.success(f"{len(ready)} өдрийн тооллого амжилттай архивлагдлаа!")
                    st.rerun()
            else:
                st.info("Тулгагдаж бэлэн болсон өдөр алга. Дээрх статусыг шалгана уу "
                        "(огноо тохирохгүй байгаа, эсвэл файл унших алдаатай байж болзошгүй).")

    history = load_json(PATH_HISTORY, [])
    deleted = load_json(PATH_DELETED, [])

    if not history:
        st.info("Одоогоор архивласан тооллого алга байна.")
    else:
        hist_df_meta = pd.DataFrame([
            {"id": r["id"], "Огноо": r["date"], "Архивласан": r.get("archived_at", ""),
             "Мөрийн тоо": len(r.get("items", []))}
            for r in history
        ])
        hist_df_meta["Сар"] = pd.to_datetime(hist_df_meta["Огноо"], errors="coerce").dt.strftime("%Y-%m")

        months = ["Бүгд"] + sorted(hist_df_meta["Сар"].dropna().unique().tolist(), reverse=True)
        sel_month = st.selectbox("📅 Сараар шүүх", months)

        filtered_meta = hist_df_meta if sel_month == "Бүгд" else hist_df_meta[hist_df_meta["Сар"] == sel_month]

        st.dataframe(filtered_meta[["Огноо", "Архивласан", "Мөрийн тоо"]], use_container_width=True, hide_index=True)

        record_options = {f'{r["date"]} — {r["id"][:8]}': r["id"] for r in history
                           if sel_month == "Бүгд" or str(r["date"])[:7] == sel_month}

        if record_options:
            sel_label = st.selectbox("Дэлгэрэнгүй харах тайлан сонгох", list(record_options.keys()))
            sel_id = record_options[sel_label]
            sel_record = next(r for r in history if r["id"] == sel_id)
            rdf = pd.DataFrame(sel_record["items"])

            st.write(f"### 🧾 {sel_record['date']} өдрийн тайлан")
            if "Зөрүү" in rdf.columns:
                st.dataframe(styled_table(rdf), use_container_width=True, hide_index=True)
            else:
                st.dataframe(rdf, use_container_width=True, hide_index=True)

            photo_paths = sel_record.get("photos", [])
            existing_photos = [p for p in photo_paths if os.path.exists(p)]
            if existing_photos:
                st.write(f"**📷 Хавсаргасан зураг ({len(existing_photos)}):**")
                photo_cols = st.columns(min(3, len(existing_photos)))
                for i, p in enumerate(existing_photos):
                    with photo_cols[i % len(photo_cols)]:
                        st.image(p, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                excel_bytes = df_to_excel_bytes(rdf, sheet_name=sel_record["date"])
                st.download_button(
                    "⬇️ Excel-ээр татах",
                    data=excel_bytes,
                    file_name=f"CaffeBene_tailan_{sel_record['date']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            with c2:
                if st.button("🗑️ Устгах (Хогийн саванд шилжүүлэх)", use_container_width=True):
                    history = [r for r in history if r["id"] != sel_id]
                    sel_record["deleted_at"] = datetime.now().isoformat()
                    deleted.append(sel_record)
                    save_json(PATH_HISTORY, history)
                    save_json(PATH_DELETED, deleted)
                    st.warning("Тайланг хогийн саванд шилжүүллээ.")
                    st.rerun()

            # Сарын нэгтгэл
            if "Зөрүү" in filtered_meta.columns or True:
                st.divider()
                st.write("### 📈 Сарын нэгтгэл")
                all_month_items = []
                for r in history:
                    if sel_month == "Бүгд" or str(r["date"])[:7] == sel_month:
                        for item in r.get("items", []):
                            item = dict(item)
                            item["Огноо"] = r["date"]
                            all_month_items.append(item)
                if all_month_items:
                    month_df = pd.DataFrame(all_month_items)
                    if "Зөрүү" in month_df.columns:
                        summary = month_df.groupby("Нэр", dropna=False).agg(
                            Бодит=("Бодит", "sum"),
                            Систем=("Систем", "sum"),
                            Зөрүү=("Зөрүү", "sum"),
                        ).reset_index()
                        st.dataframe(styled_table(summary), use_container_width=True, hide_index=True)
                        month_excel = df_to_excel_bytes(summary, sheet_name="Сарын_нэгтгэл")
                        st.download_button(
                            "⬇️ Сарын нэгтгэлийг Excel-ээр татах",
                            data=month_excel,
                            file_name=f"CaffeBene_saryn_negtgel_{sel_month}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

    st.divider()
    with st.expander(f"🗑️ Хогийн сав ({len(deleted)})", expanded=False):
        if not deleted:
            st.caption("Хогийн сав хоосон байна.")
        else:
            for r in deleted:
                cols = st.columns([3, 2, 2])
                cols[0].write(f"**{r['date']}** — {r['id'][:8]}")
                cols[1].write(f"Устгасан: {r.get('deleted_at', '—')}")
                if cols[2].button("♻️ Сэргээх", key=f"restore_{r['id']}"):
                    deleted = [d for d in deleted if d["id"] != r["id"]]
                    r.pop("deleted_at", None)
                    history.append(r)
                    save_json(PATH_HISTORY, history)
                    save_json(PATH_DELETED, deleted)
                    st.success("Тайланг сэргээлээ.")
                    st.rerun()


# =======================================================================================
# SIDEBAR — Товч заавар
# =======================================================================================
with st.sidebar:
    st.markdown("### ☕ Caffe Bene")
    st.caption("Тооллого & Тулгалтын систем")
    st.markdown("---")
    st.markdown(
        """
        **Ажиллах дараалал:**
        1. 📝 **ТООЛЛОГО** — Ø/Х/О тоог шивнэ, эсвэл Excel/зургаас ачаална.
        2. Системийн Excel (`Qty Sold`) upload хийж **Тулгалт хийх**.
        3. Зөрүүг шалгаад **Архивлах**.
        4. 📊 **АРХИВ** — сараар харах, Excel татах, устгах/сэргээх.
        5. Бүтэн сарын файлуудыг **АРХИВ** табны багц импортоор нэг дор оруулж болно.
        """
    )
    st.markdown("---")
    st.caption(f"Өгөгдлийн сан: `{DATA_DIR}/`")
    st.caption("© Caffe Bene — Дотоод хэрэглээний систем")
