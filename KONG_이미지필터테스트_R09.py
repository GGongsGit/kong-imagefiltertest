import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import configparser
import os
import sys
import tempfile

CURRENT_IMAGE_PATH = None

# 응용 방법별 최적 조합 INI 파일
PIPELINE_INI_PATH = "pipelines.ini"


# =========================================================
# 0) 유니코드/특수문자/깨진 문자열 경로 대응 유틸
# =========================================================
def sanitize_path(p):
    """
    - bytes로 들어온 경우도 안전하게 str로 변환
    - 앞뒤 공백 제거
    - Windows 경로 구분자 통일( \\ -> / )
    """
    if p is None:
        return ""

    # bytes -> str (깨진 인코딩 가능성 완화)
    if isinstance(p, (bytes, bytearray)):
        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                p = p.decode(enc)
                break
            except Exception:
                continue
        else:
            p = p.decode("utf-8", errors="replace")

    p = str(p).strip()
    p = p.replace("\\", "/")
    return p


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """
    한글/공백/특수문자 경로에서도 안정적으로 이미지 읽기 (Windows에서 특히 유효)
    """
    path = sanitize_path(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)   # 유니코드 경로 OK
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def acquire_single_instance_lock(app_id="opencv_cheatsheet_app.lock"):
    """
    이미 실행 중이면 None 반환.
    최초 실행이면 lock 파일 핸들 반환(프로그램 종료까지 유지 필요).
    """
    lock_path = os.path.join(tempfile.gettempdir(), app_id)

    try:
        # O_EXCL: 이미 있으면 실패 -> 중복 실행 감지
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        return fd
    except FileExistsError:
        return None
    except Exception:
        return None


# -------------------------------
# 1) 이미지 분석 함수
# -------------------------------
def analyze_image(gray):
    h, w = gray.shape[:2]

    mean_brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges)) / (h * w)

    blocks_y, blocks_x = 4, 4
    block_means = []
    for by in range(blocks_y):
        for bx in range(blocks_x):
            y0 = by * h // blocks_y
            y1 = (by + 1) * h // blocks_y
            x0 = bx * w // blocks_x
            x1 = (bx + 1) * w // blocks_x
            block = gray[y0:y1, x0:x1]
            block_means.append(np.mean(block))
    block_means = np.array(block_means, dtype=np.float32)
    illum_uneven = float(block_means.std())

    blur_for_noise = cv2.GaussianBlur(gray, (5, 5), 0)
    high_freq = cv2.absdiff(gray, blur_for_noise)
    noise_level = float(np.std(high_freq))

    stats = {
        "mean_brightness": mean_brightness,
        "contrast": contrast,
        "lap_var": lap_var,
        "edge_density": edge_density,
        "illum_uneven": illum_uneven,
        "noise_level": noise_level,
    }
    return stats


# -------------------------------
# 2) 이미지 유형 추정
# -------------------------------
def classify_image(stats):
    mean_b = stats["mean_brightness"]
    contrast = stats["contrast"]
    lap_var = stats["lap_var"]
    edge_density = stats["edge_density"]
    illum_uneven = stats["illum_uneven"]
    noise_level = stats["noise_level"]

    is_dark = mean_b < 80
    is_low_contrast = contrast < 40
    is_very_blurry = lap_var < 80
    is_blurry = lap_var < 150
    is_noisy = noise_level > 25
    is_uneven_light = illum_uneven > 20
    is_edge_rich = edge_density > 0.12

    if not is_dark and is_edge_rich and not is_very_blurry:
        if is_uneven_light:
            return "doc_uneven"
        else:
            return "doc_clean"

    if is_dark or is_low_contrast:
        return "low_light"

    if is_very_blurry:
        return "very_blurry"

    if is_noisy and is_edge_rich:
        return "noisy_texture"

    if is_edge_rich and not is_blurry:
        return "object_edge"

    return "generic"


# -------------------------------
# 3) 시나리오별 파이프라인
# -------------------------------
def get_pipeline(scenario):
    if scenario == "doc_clean":
        return {
            "name": "문서/명함(OCR)용 - 균일 조명",
            "steps": [
                {"name": "gaussian_blur", "params": {"ksize": (3, 3), "sigmaX": 0}, "description": "미세 노이즈 제거(GaussianBlur 3x3)"},
                {"name": "adaptive_thresh", "params": {"method": "gaussian", "block_size": 31, "C": 10}, "description": "조명 변화에 강한 AdaptiveThreshold(Gaussian)"},
                {"name": "morph_open", "params": {"ksize": (3, 3), "iterations": 1}, "description": "점 노이즈 제거(Opening)"},
                {"name": "morph_close", "params": {"ksize": (3, 3), "iterations": 1}, "description": "문자 내부 빈틈 메우기(Closing)"},
            ],
        }

    if scenario == "doc_uneven":
        return {
            "name": "문서/명함(OCR)용 - 조명 불균일",
            "steps": [
                {"name": "bilateral", "params": {"d": 9, "sigmaColor": 75, "sigmaSpace": 75}, "description": "엣지 보존 노이즈 제거(Bilateral)"},
                {"name": "adaptive_thresh", "params": {"method": "gaussian", "block_size": 35, "C": 5}, "description": "조명 변화에 강한 AdaptiveThreshold"},
                {"name": "morph_close", "params": {"ksize": (3, 3), "iterations": 2}, "description": "문서 테두리/문자 영역 보강(Closing)"},
            ],
        }

    if scenario == "low_light":
        return {
            "name": "저조도/낮은 대비 영상용",
            "steps": [
                {"name": "clahe", "params": {"clipLimit": 2.0, "tileGridSize": (8, 8)}, "description": "CLAHE로 대비 향상"},
                {"name": "gaussian_blur", "params": {"ksize": (3, 3), "sigmaX": 0}, "description": "노이즈 완화(GaussianBlur)"},
                {"name": "canny", "params": {"threshold1": 50, "threshold2": 150}, "description": "엣지 추출(Canny)"},
            ],
        }

    if scenario == "very_blurry":
        return {
            "name": "블러(흐림) 심한 이미지용",
            "steps": [
                {"name": "unsharp", "params": {"alpha": 1.5, "beta": -0.5}, "description": "Unsharp Mask로 엣지 강조"},
                {"name": "gaussian_blur", "params": {"ksize": (3, 3), "sigmaX": 0}, "description": "샤프닝 후 약한 블러로 노이즈 완화"},
                {"name": "canny", "params": {"threshold1": 80, "threshold2": 180}, "description": "엣지 추출(Canny)"},
            ],
        }

    if scenario == "noisy_texture":
        return {
            "name": "노이즈 많은 텍스처/산업용 이미지용",
            "steps": [
                {"name": "median_blur", "params": {"ksize": 5}, "description": "소금/후추 노이즈 제거(MedianBlur)"},
                {"name": "gaussian_blur", "params": {"ksize": (3, 3), "sigmaX": 0}, "description": "잔여 고주파 노이즈 제거(GaussianBlur)"},
                {"name": "canny", "params": {"threshold1": 80, "threshold2": 200}, "description": "경계선 추출(Canny)"},
                {"name": "morph_close", "params": {"ksize": (3, 3), "iterations": 1}, "description": "끊어진 엣지 연결(Closing)"},
            ],
        }

    if scenario == "object_edge":
        return {
            "name": "일반 물체 엣지/윤곽 추출용",
            "steps": [
                {"name": "gaussian_blur", "params": {"ksize": (5, 5), "sigmaX": 0}, "description": "전처리 GaussianBlur(5x5)"},
                {"name": "canny", "params": {"threshold1": 70, "threshold2": 180}, "description": "윤곽선 추출(Canny)"},
                {"name": "morph_close", "params": {"ksize": (3, 3), "iterations": 1}, "description": "끊긴 경계 보강(Closing)"},
            ],
        }

    return {
        "name": "기본(Generic) 전처리",
        "steps": [
            {"name": "gaussian_blur", "params": {"ksize": (5, 5), "sigmaX": 0}, "description": "일반적 노이즈 제거(GaussianBlur)"},
            {"name": "canny", "params": {"threshold1": 100, "threshold2": 200}, "description": "기본 엣지 추출(Canny)"},
        ],
    }


# -------------------------------
# 4) 파이프라인 적용
# -------------------------------
def apply_pipeline(gray, pipeline):
    result = gray.copy()

    for step in pipeline["steps"]:
        name = step["name"]
        p = step["params"]

        if name == "gaussian_blur":
            result = cv2.GaussianBlur(result, p["ksize"], p["sigmaX"])
        elif name == "median_blur":
            result = cv2.medianBlur(result, p["ksize"])
        elif name == "bilateral":
            result = cv2.bilateralFilter(result, p["d"], p["sigmaColor"], p["sigmaSpace"])
        elif name == "adaptive_thresh":
            m = cv2.ADAPTIVE_THRESH_MEAN_C if p["method"] == "mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            result = cv2.adaptiveThreshold(result, 255, m, cv2.THRESH_BINARY, p["block_size"], p["C"])
        elif name == "morph_open":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, p["ksize"])
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=p["iterations"])
        elif name == "morph_close":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, p["ksize"])
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=p["iterations"])
        elif name == "canny":
            result = cv2.Canny(result, p["threshold1"], p["threshold2"])
        elif name == "clahe":
            clahe = cv2.createCLAHE(clipLimit=p["clipLimit"], tileGridSize=p["tileGridSize"])
            result = clahe.apply(result)
        elif name == "unsharp":
            blurred = cv2.GaussianBlur(result, (0, 0), 3)
            result = cv2.addWeighted(result, p["alpha"], blurred, p["beta"], 0)

    return result


# -------------------------------
# 5) 샘플 코드 자동 생성 (Tab1)
# -------------------------------
def generate_sample_code(pipeline, image_path):
    image_path_str = sanitize_path(image_path)

    lines = []
    lines.append("import cv2")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("def imread_unicode(path, flags=cv2.IMREAD_COLOR):")
    lines.append("    path = str(path)")
    lines.append("    data = np.fromfile(path, dtype=np.uint8)")
    lines.append("    img = cv2.imdecode(data, flags)")
    lines.append("    return img")
    lines.append("")
    lines.append(f'img = imread_unicode(r"{image_path_str}")')
    lines.append("if img is None:")
    lines.append('    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")')
    lines.append("gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
    lines.append("")
    lines.append(f"# --- Recommended pipeline: {pipeline['name']} ---")
    lines.append("result = gray.copy()")
    lines.append("")

    for step in pipeline["steps"]:
        name = step["name"]
        p = step["params"]

        if name == "gaussian_blur":
            lines.append("# Gaussian Blur")
            lines.append(f"result = cv2.GaussianBlur(result, {p['ksize']}, {p['sigmaX']})")
        elif name == "median_blur":
            lines.append("# Median Blur")
            lines.append(f"result = cv2.medianBlur(result, {p['ksize']})")
        elif name == "bilateral":
            lines.append("# Bilateral Filter")
            lines.append(f"result = cv2.bilateralFilter(result, d={p['d']}, sigmaColor={p['sigmaColor']}, sigmaSpace={p['sigmaSpace']})")
        elif name == "adaptive_thresh":
            meth_str = "cv2.ADAPTIVE_THRESH_MEAN_C" if p["method"] == "mean" else "cv2.ADAPTIVE_THRESH_GAUSSIAN_C"
            lines.append("# Adaptive Threshold")
            lines.append(
                "result = cv2.adaptiveThreshold(\n"
                "    result,\n"
                "    255,\n"
                f"    {meth_str},\n"
                "    cv2.THRESH_BINARY,\n"
                f"    {p['block_size']},\n"
                f"    {p['C']},\n"
                ")"
            )
        elif name == "morph_open":
            lines.append("# Morphology Opening")
            lines.append(f"kernel = cv2.getStructuringElement(cv2.MORPH_RECT, {p['ksize']})")
            lines.append(f"result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations={p['iterations']})")
        elif name == "morph_close":
            lines.append("# Morphology Closing")
            lines.append(f"kernel = cv2.getStructuringElement(cv2.MORPH_RECT, {p['ksize']})")
            lines.append(f"result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations={p['iterations']})")
        elif name == "canny":
            lines.append("# Canny")
            lines.append(f"result = cv2.Canny(result, {p['threshold1']}, {p['threshold2']})")
        elif name == "clahe":
            lines.append("# CLAHE")
            lines.append(f"clahe = cv2.createCLAHE(clipLimit={p['clipLimit']}, tileGridSize={p['tileGridSize']})")
            lines.append("result = clahe.apply(result)")
        elif name == "unsharp":
            lines.append("# Unsharp Mask")
            lines.append("blurred = cv2.GaussianBlur(result, (0, 0), 3)")
            lines.append(f"result = cv2.addWeighted(result, {p['alpha']}, blurred, {p['beta']}, 0)")

        lines.append("")

    lines.append("# GUI 없이 단독 실행 시에는 cv2.imshow()/waitKey()를 사용하시면 됩니다.")
    return "\n".join(lines)


# -------------------------------
# 6) Tab1: 자동 분석 앱
# -------------------------------
class AutoFilterApp:
    def __init__(self, parent):
        self.master = parent
        self.image_path_var = tk.StringVar()
        padding = {"padx": 10, "pady": 5}

        lbl_title = ttk.Label(self.master, text="이미지 자동 분석 & 최적 필터 조합 추천", font=("맑은 고딕", 12, "bold"))
        lbl_title.pack(pady=10, anchor="n")

        frm_top = ttk.Frame(self.master)
        frm_top.pack(fill="x", **padding)

        ttk.Label(frm_top, text="이미지 파일 경로:").pack(side="left")
        self.entry_path = ttk.Entry(frm_top, textvariable=self.image_path_var)
        self.entry_path.pack(side="left", fill="x", expand=True, padx=5)

        ttk.Button(frm_top, text="찾아보기...", command=self.browse_image).pack(side="left")
        ttk.Button(frm_top, text="분석 및 처리 실행", command=self.run_analysis).pack(side="left", padx=5)

        frm_images = ttk.Frame(self.master)
        frm_images.pack(fill="both", expand=False, **padding)

        frm_orig = ttk.LabelFrame(frm_images, text="원본 이미지")
        frm_orig.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.lbl_orig_img = ttk.Label(frm_orig)
        self.lbl_orig_img.pack(fill="both", expand=True)

        frm_res = ttk.LabelFrame(frm_images, text="필터 적용 결과")
        frm_res.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.lbl_res_img = ttk.Label(frm_res)
        self.lbl_res_img.pack(fill="both", expand=True)

        frm_bottom = ttk.Frame(self.master)
        frm_bottom.pack(fill="both", expand=True, **padding)

        frm_tables = ttk.Frame(frm_bottom)
        frm_tables.pack(side="left", fill="both", expand=False, padx=5)

        ttk.Label(frm_tables, text="분석 결과(표)").pack(anchor="w")

        self.tree_stats = ttk.Treeview(frm_tables, columns=("metric", "value"), show="headings", height=6)
        self.tree_stats.heading("metric", text="지표")
        self.tree_stats.heading("value", text="값")
        self.tree_stats.column("metric", width=130, anchor="center")
        self.tree_stats.column("value", width=120, anchor="e")
        self.tree_stats.pack(fill="x", pady=(0, 10))

        ttk.Label(frm_tables, text="추천 필터 파이프라인(표)").pack(anchor="w")

        self.tree_steps = ttk.Treeview(frm_tables, columns=("idx", "step", "desc"), show="headings", height=10)
        self.tree_steps.heading("idx", text="순서")
        self.tree_steps.heading("step", text="필터")
        self.tree_steps.heading("desc", text="설명")
        self.tree_steps.column("idx", width=40, anchor="center")
        self.tree_steps.column("step", width=120, anchor="center")
        self.tree_steps.column("desc", width=260, anchor="w")
        self.tree_steps.pack(fill="both", expand=True)

        frm_code = ttk.Frame(frm_bottom)
        frm_code.pack(side="left", fill="both", expand=True, padx=5)

        ttk.Label(frm_code, text="이 파이프라인을 적용하는 샘플 코드:").pack(anchor="w")
        self.txt_result = tk.Text(frm_code, height=20, font=("Consolas", 9))
        self.txt_result.pack(fill="both", expand=True)

    def show_image_on_label(self, img_bgr_or_gray, label_widget, max_size=(450, 320)):
        if img_bgr_or_gray is None:
            return

        if len(img_bgr_or_gray.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        max_w, max_h = max_size
        scale = min(max_w / w, max_h / h, 1.0)
        if scale != 1.0:
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk

    def browse_image(self):
        global CURRENT_IMAGE_PATH
        filetypes = [
            ("이미지 파일", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"),
            ("모든 파일", "*.*"),
        ]
        filepath = filedialog.askopenfilename(title="이미지 파일 선택", filetypes=filetypes)
        if filepath:
            filepath = sanitize_path(filepath)              # ✅ 경로 정리
            self.image_path_var.set(filepath)
            CURRENT_IMAGE_PATH = filepath                   # ✅ 그대로 저장(이미 정리됨)

            img = imread_unicode(filepath, cv2.IMREAD_COLOR)
            if img is not None:
                self.show_image_on_label(img, self.lbl_orig_img)

    def run_analysis(self):
        global CURRENT_IMAGE_PATH
        try:
            path_str = sanitize_path(self.image_path_var.get())
            if not path_str:
                messagebox.showwarning("경고", "이미지 파일 경로를 먼저 선택해 주세요.")
                return

            img_path = Path(path_str)
            if not img_path.exists():
                messagebox.showerror("에러", f"파일을 찾을 수 없습니다:\n{img_path}")
                return

            img = imread_unicode(img_path, cv2.IMREAD_COLOR)
            if img is None:
                messagebox.showerror("에러", "이미지를 읽을 수 없습니다. 경로와 파일명을 확인해 주세요.")
                return

            CURRENT_IMAGE_PATH = sanitize_path(img_path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            stats = analyze_image(gray)
            scenario = classify_image(stats)
            pipeline = get_pipeline(scenario)
            result = apply_pipeline(gray, pipeline)

            for row in self.tree_stats.get_children():
                self.tree_stats.delete(row)

            label_map = {
                "mean_brightness": "평균 밝기",
                "contrast": "대비(표준편차)",
                "lap_var": "라플라시안 분산(초점)",
                "edge_density": "엣지 밀도",
                "illum_uneven": "조명 불균일(분산)",
                "noise_level": "노이즈 수준",
            }
            for key, value in stats.items():
                self.tree_stats.insert("", "end", values=(label_map.get(key, key), f"{value:.3f}"))

            for row in self.tree_steps.get_children():
                self.tree_steps.delete(row)

            self.tree_steps.insert("", "end", values=(" ", f"[시나리오] {scenario}", pipeline["name"]))
            for i, step in enumerate(pipeline["steps"], 1):
                self.tree_steps.insert("", "end", values=(i, step["name"], step["description"]))

            self.txt_result.delete("1.0", tk.END)
            self.txt_result.insert(tk.END, generate_sample_code(pipeline, img_path))

            self.show_image_on_label(img, self.lbl_orig_img, max_size=(450, 320))
            self.show_image_on_label(result, self.lbl_res_img, max_size=(450, 320))

        except Exception as e:
            messagebox.showerror("분석 실행 오류", str(e))
            raise


# -------------------------------
# 7) INI 관리: 응용 필터 조합 로드/저장
# -------------------------------
def ensure_pipeline_ini_exists():
    if os.path.exists(PIPELINE_INI_PATH):
        return
    cfg = configparser.ConfigParser()
    cfg["meta"] = {"version": "1"}
    defaults = [
        ("문서/명함 OCR(조명 균일)", "선명한 텍스트 이진화", "GaussianBlur(3×3) → AdaptiveThreshold(Gaussian) → Opening → Closing"),
        ("문서/명함+조명 불균일", "문서 테두리/텍스트 안정적 추출", "Bilateral → AdaptiveThreshold → Closing → Canny(낮은 T) → Contour"),
        ("RealSense/L515 PnP", "안정적인 엣지+Depth ROI", "GaussianBlur(5×5) → Canny → Depth Mask → Closing"),
        ("금속/PCB 고노이즈", "반사/노이즈 줄인 엣지", "MedianBlur(5~7) → GaussianBlur(3×3) → Canny → Closing"),
        ("흐린 이미지 Edge", "흐린 경계 보존", "Unsharp Mask → GaussianBlur(3×3) → Canny"),
        ("저조도/낮은 대비", "대비/밝기 향상+엣지 안정화", "CLAHE → GaussianBlur → Canny"),
        ("객체 분리/배경 제거", "깨끗한 객체 마스크", "GaussianBlur(7×7) → Otsu Threshold → Closing → Largest Contour"),
    ]
    for i, (sc, goal, pipe) in enumerate(defaults, 1):
        sec = f"pipeline_{i:03d}"
        cfg[sec] = {"scenario": sc, "goal": goal, "pipeline": pipe}

    with open(PIPELINE_INI_PATH, "w", encoding="utf-8") as f:
        cfg.write(f)


def load_pipelines_from_ini():
    ensure_pipeline_ini_exists()
    cfg = configparser.ConfigParser()
    cfg.read(PIPELINE_INI_PATH, encoding="utf-8")

    items = []
    for sec in cfg.sections():
        if sec == "meta":
            continue
        sc = cfg.get(sec, "scenario", fallback="").strip()
        goal = cfg.get(sec, "goal", fallback="").strip()
        pipe = cfg.get(sec, "pipeline", fallback="").strip()
        if sc and goal and pipe:
            items.append((sec, sc, goal, pipe))
    return items


def add_pipeline_to_ini(scenario, goal, pipeline_text):
    ensure_pipeline_ini_exists()
    cfg = configparser.ConfigParser()
    cfg.read(PIPELINE_INI_PATH, encoding="utf-8")

    existing = [s for s in cfg.sections() if s.startswith("pipeline_")]
    if existing:
        nums = []
        for s in existing:
            try:
                nums.append(int(s.split("_")[1]))
            except Exception:
                pass
        next_num = (max(nums) + 1) if nums else 1
    else:
        next_num = 1

    sec = f"pipeline_{next_num:03d}"
    cfg[sec] = {"scenario": scenario, "goal": goal, "pipeline": pipeline_text}

    with open(PIPELINE_INI_PATH, "w", encoding="utf-8") as f:
        cfg.write(f)

    return sec


# -------------------------------
# 8) Tab2: 치트시트/샘플코드 탭
# -------------------------------
def build_cheatsheet_tab(parent):
    global CURRENT_IMAGE_PATH

    paned_lr = ttk.Panedwindow(parent, orient="horizontal")
    paned_lr.pack(fill="both", expand=True, padx=30, pady=30)

    left_frame = ttk.Frame(paned_lr)
    paned_lr.add(left_frame, weight=3)

    right_frame = ttk.Frame(paned_lr)
    paned_lr.add(right_frame, weight=4)

    lf = ttk.LabelFrame(left_frame, text="OpenCV 분류 체계 치트시트")
    lf.pack(fill="both", expand=True)

    tree_tax = ttk.Treeview(
        lf,
        columns=("no", "group", "type", "func_name", "func_desc"),
        show="headings",
        height=28
    )
    tree_tax.heading("no", text="No")
    tree_tax.heading("group", text="구분")
    tree_tax.heading("type", text="종류")
    tree_tax.heading("func_name", text="OpenCV 함수")
    tree_tax.heading("func_desc", text="주요 기능")

    tree_tax.column("no", width=40, anchor="center", stretch=False)
    tree_tax.column("group", width=60, anchor="center", stretch=False)
    tree_tax.column("type", width=140, anchor="w")
    tree_tax.column("func_name", width=190, anchor="w")

    vsb = ttk.Scrollbar(lf, orient="vertical", command=tree_tax.yview)
    tree_tax.configure(yscrollcommand=vsb.set)
    tree_tax.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    taxonomy = [
        (1, "픽셀 단위(Pixel Ops)", [
            ("Gamma correction", "cv2.LUT (권장) / 직접 구현", "밝기 비선형 보정, 저조도/과노출 보정"),
            ("Histogram equalization", "cv2.equalizeHist", "전체 대비 향상(그레이)"),
            ("CLAHE", "cv2.createCLAHE().apply", "국부 대비 향상, 과증폭 방지"),
            ("Averaging", "cv2.add / cv2.addWeighted", "단순 픽셀 결합(밝기/블렌딩)"),
            ("Normalization", "cv2.normalize", "값 범위 정규화"),
            ("Logical Ops", "cv2.bitwise_and/or/xor/not", "AND/OR/XOR, 마스크 처리"),
            ("convertScaleAbs", "cv2.convertScaleAbs", "미분 결과를 8bit 영상으로 변환"),
        ]),
        (2, "Filtering&Noise reduction(필터링)", [
            ("Convolution", "cv2.filter2D", "모든 필터의 기본 연산"),
            ("Average filter", "cv2.blur / cv2.boxFilter", "랜덤 노이즈 제거"),
            ("Gaussian filter", "cv2.GaussianBlur", "고주파 노이즈 제거, Canny 전처리"),
            ("Median filter", "cv2.medianBlur", "소금·후추 노이즈 제거"),
            ("Bilateral filter", "cv2.bilateralFilter", "엣지 보존 + 노이즈 제거"),
            ("Non-local Means", "cv2.fastNlMeansDenoising(Colored)", "텍스처 유지 고급 노이즈 제거"),
        ]),
        (3, "Gradient & Edges Detection (미분 기반)", [
            ("Prewitt", "filter2D(커널 구현)", "1차 미분, 고전 연산자"),
            ("Sobel", "cv2.Sobel", "방향성 엣지 검출"),
            ("Scharr", "cv2.Scharr", "Sobel 고정밀 버전"),
            ("Robert", "filter2D(커널 구현)", "대각선 민감"),
            ("Laplacian", "cv2.Laplacian", "2차 미분, 전체 엣지"),
            ("Canny", "cv2.Canny", "표준 엣지 검출 (NMS+Hysteresis)"),
        ]),
        (4, "Morphological (형태학)", [
            ("Dilation (팽창)", "cv2.dilate", "영역 확장/구멍 메우기"),
            ("Erosion (침식)", "cv2.erode", "작은 흰색 노이즈 제거"),
            ("Opening (Erosion → Dilation)", "cv2.morphologyEx(MORPH_OPEN)", "점 노이즈 제거"),
            ("Closing (Dilation → Erosion)", "cv2.morphologyEx(MORPH_CLOSE)", "내부 구멍 메우기"),
            ("Morph Gradient", "cv2.morphologyEx(MORPH_GRADIENT)", "경계선 추출"),
            ("Skeletonization", "반복 erode/subtract 구현", "중심선 추출(직접 구현)"),
            ("Distance transform", "cv2.distanceTransform", "객체 중심 거리 계산"),
            ("Watershed", "cv2.watershed", "영역 분할(Segmentation)"),
        ]),
        (5, "Thresholding & Segmentation", [
            ("Global Threshold", "cv2.threshold", "조명 균일"),
            ("Adaptive Threshold", "cv2.adaptiveThreshold", "조명 불균일"),
            ("Otsu Threshold", "cv2.threshold(..., THRESH_OTSU)", "자동 임계값"),
            ("Connected Components", "cv2.connectedComponentsWithStats", "객체 분리"),
            ("Grabcut", "cv2.grabCut", "전경/배경 분리"),
        ]),
        (6, "Feature Detection & Description", [
            ("Harris Corner", "cv2.cornerHarris", "코너 검출"),
            ("Shi-Tomasi", "cv2.goodFeaturesToTrack", "안정적인 코너"),
            ("FAST", "cv2.FastFeatureDetector_create", "고속 코너 검출"),
            ("ORB", "cv2.ORB_create", "회전/스케일 불변 특징"),
            ("SIFT", "cv2.SIFT_create", "고정밀 특징"),
            ("SURF", "cv2.xfeatures2d.SURF_create", "Contrib(환경 의존)"),
            ("BRIEF", "cv2.xfeatures2d.BriefDescriptorExtractor_create", "Contrib(경량 디스크립터)"),
        ]),
        (7, "Shape & Line Detection", [
            ("Contours", "cv2.findContours / cv2.drawContours", "객체 외곽선"),
            ("Hough Line", "cv2.HoughLines / cv2.HoughLinesP", "직선 검출"),
            ("Hough Circle", "cv2.HoughCircles", "원 검출"),
            ("Convex Hull", "cv2.convexHull", "외곽 다각형"),
            ("Bounding Box", "cv2.boundingRect / cv2.minAreaRect", "객체 영역"),
        ]),
        (8, "Geometric\\Transform", [
            ("Translation", "cv2.warpAffine", "평행 이동(아핀 행렬 적용)"),
            ("Rotation", "cv2.getRotationMatrix2D / cv2.warpAffine", "회전"),
            ("Scale", "cv2.resize", "크기 변경"),
            ("Affine", "cv2.getAffineTransform / cv2.warpAffine", "회전+이동+스케일"),
            ("Similarity", "warpAffine(유사변환 행렬)", "비율 유지 변환"),
            ("Projective", "cv2.getPerspectiveTransform / cv2.warpPerspective", "원근 보정"),
            ("Rigid (Euclidean)", "warpAffine(회전+이동만)", "회전+이동만"),
        ]),
        (9, "Interpolation & Resampling", [
            ("Nearest", "cv2.resize(INTER_NEAREST)", "빠름, 품질 낮음"),
            ("Bilinear", "cv2.resize(INTER_LINEAR)", "기본"),
            ("Bicubic", "cv2.resize(INTER_CUBIC)", "고품질"),
            ("Area", "cv2.resize(INTER_AREA)", "축소 시 최적"),
        ]),
        (10, "Frequency Domain", [
            ("FFT", "np.fft.fft2 / np.fft.fftshift", "주파수 분석"),
            ("DCT", "cv2.dct", "압축(JPEG)"),
            ("High-pass", "FFT 후 마스크링", "엣지 강조"),
            ("Low-pass", "FFT 후 마스크링", "블러"),
        ]),
        (11, "Color Space Processing", [
            ("RGB ↔ Gray", "cv2.cvtColor", "기본 변환"),
            ("HSV", "cv2.cvtColor(BGR2HSV)", "색 기반 분할"),
            ("Lab", "cv2.cvtColor(BGR2Lab)", "조명 영향 최소"),
            ("YCrCb", "cv2.cvtColor(BGR2YCrCb)", "피부/객체 분리"),
        ]),
    ]

    for no, group, items in taxonomy:
        first = True
        for t, func_name, func_desc in items:
            tree_tax.insert("", "end", values=(str(no) if first else "", group if first else "", t, func_name, func_desc))
            first = False

    paned_tb = ttk.Panedwindow(right_frame, orient="vertical")
    paned_tb.pack(fill="both", expand=True)

    top_right = ttk.Frame(paned_tb)
    bottom_right = ttk.Frame(paned_tb)
    paned_tb.add(top_right, weight=4)
    paned_tb.add(bottom_right, weight=3)

    tr = ttk.LabelFrame(top_right, text="선택한 연산의 Python 샘플 코드 (수정 가능)")
    tr.pack(fill="both", expand=True)

    code_preview = ttk.Panedwindow(tr, orient="horizontal")
    code_preview.pack(fill="both", expand=True, padx=8, pady=8)

    code_area = ttk.Frame(code_preview)
    preview_area = ttk.Frame(code_preview)
    code_preview.add(code_area, weight=2)
    code_preview.add(preview_area, weight=6)

    txt_code = tk.Text(code_area, height=14, font=("Consolas", 9))
    txt_code.pack(fill="both", expand=True)

    btn_run = ttk.Button(code_area, text="코드 실행 및 결과 표시")
    btn_run.pack(anchor="e", pady=(6, 0))

    ttk.Label(preview_area, text="실행 결과 미리보기").pack(anchor="w")
    lbl_preview = ttk.Label(preview_area, relief="sunken", anchor="center")
    lbl_preview.pack(fill="both", expand=True, pady=6)

    _preview_state = {"img_rgb": None}

    def _render_preview():
        img_rgb = _preview_state["img_rgb"]
        if img_rgb is None:
            return

        box_w = max(lbl_preview.winfo_width(), 1)
        box_h = max(lbl_preview.winfo_height(), 1)

        h, w = img_rgb.shape[:2]
        if w <= 0 or h <= 0:
            return

        scale = min(box_w / w, box_h / h)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=interp)

        img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        lbl_preview.configure(image=img_tk)
        lbl_preview.image = img_tk

    def show_result(img_bgr_or_gray):
        if img_bgr_or_gray is None:
            return

        if len(img_bgr_or_gray.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)

        _preview_state["img_rgb"] = img_rgb
        _render_preview()

    def _on_preview_resize(event):
        _render_preview()

    lbl_preview.bind("<Configure>", _on_preview_resize)

    def print_cheatsheet_widget_sizes():
        try:
            code_w = txt_code.winfo_width()
            code_h = txt_code.winfo_height()
            img_w = lbl_preview.winfo_width()
            img_h = lbl_preview.winfo_height()
            # print(
            #     f"[CheatSheet Size]\n"
            #     f"  - Source Code Text : {code_w} x {code_h}\n"
            #     f"  - Image Preview    : {img_w} x {img_h}\n"
            # )
        except Exception as e:
            print("[Size Print Error]", e)

    # ---------------------------------------------------------
    # ✅ 모든 샘플코드에서 cv2.imread -> imread_unicode 사용
    # ---------------------------------------------------------
    snippet_by_type = {
        "Gaussian filter": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.GaussianBlur(img, (5, 5), 0)
show_result(out)
""",
        "Median filter": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.medianBlur(img, 5)
show_result(out)
""",
        "Bilateral filter": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
show_result(out)
""",
        "Global Threshold": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
show_result(binary)
""",
        "Adaptive Threshold": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

binary = cv2.adaptiveThreshold(
    img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31, 10
)
show_result(binary)
""",
        "Otsu Threshold": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show_result(binary)
""",
        "Canny": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

blur = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blur, 70, 180)
show_result(edges)
""",
        "Sobel": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

absx = cv2.convertScaleAbs(sx)
absy = cv2.convertScaleAbs(sy)
grad = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

show_result(grad)
""",
        "Laplacian": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
out = cv2.convertScaleAbs(lap)
show_result(out)
""",
        "Contours": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

show_result(img)
""",
        "Hough Line": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=50, maxLineGap=10)
if lines is not None:
    for x1, y1, x2, y2 in lines[:,0,:]:
        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

show_result(img)
""",
        "Hough Circle": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=30, minRadius=5, maxRadius=200)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for x, y, r in circles[0, :]:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)

show_result(img)
""",
        "ORB": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

orb = cv2.ORB_create()
kps, des = orb.detectAndCompute(img, None)

out = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_result(out)
""",
        "SIFT": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

sift = cv2.SIFT_create()
kps, des = sift.detectAndCompute(img, None)

out = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_result(out)
""",
        "Harris Corner": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f = np.float32(gray)

dst = cv2.cornerHarris(gray_f, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]
show_result(img)
""",
        "Dilation (팽창)": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
out = cv2.dilate(img, kernel, iterations=1)
show_result(out)
""",
        "Erosion (침식)": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
out = cv2.erode(img, kernel, iterations=1)
show_result(out)
""",
        "convertScaleAbs": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
out = cv2.convertScaleAbs(sx)
show_result(out)
""",
        "CLAHE": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
out = clahe.apply(img)
show_result(out)
""",
        "Histogram equalization": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.equalizeHist(img)
show_result(out)
""",
        "RGB ↔ Gray": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_result(gray)
""",
        "HSV": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
show_result(hsv)
""",
        "Projective": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape[:2]
src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
dst = np.float32([[20,20],[w-40,10],[w-20,h-30],[30,h-10]])

M = cv2.getPerspectiveTransform(src, dst)
out = cv2.warpPerspective(img, M, (w, h))
show_result(out)
""",
    }

    # ---------------------------------------------------------
    # (추가) 누락 샘플 코드 매핑들 (모두 imread_unicode 적용)
    # ---------------------------------------------------------
    snippet_by_type.update({
        "Gamma correction": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gamma = 1.8
inv = 1.0 / gamma
table = (np.arange(256) / 255.0) ** inv
table = np.clip(table * 255.0, 0, 255).astype(np.uint8)

out = cv2.LUT(img, table)
show_result(out)
""",
        "Averaging": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

blur = cv2.GaussianBlur(img, (9, 9), 0)
out = cv2.addWeighted(img, 0.7, blur, 0.3, 0)
show_result(out)
""",
        "Normalization": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show_result(out)
""",
        "Logical Ops": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (w//2, h//2), min(h, w)//4, 255, -1)

out = cv2.bitwise_and(img, img, mask=mask)
show_result(out)
""",
        "Convolution": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

kernel = np.array([[0,-1,0],
                   [-1, 5,-1],
                   [0,-1,0]], dtype=np.float32)

out = cv2.filter2D(img, -1, kernel)
show_result(out)
""",
        "Average filter": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.blur(img, (7, 7))
show_result(out)
""",
        "Non-local Means": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

if len(img.shape) == 3:
    out = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
else:
    out = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

show_result(out)
""",
        "Prewitt": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

kx = np.array([[-1,0,1],
               [-1,0,1],
               [-1,0,1]], dtype=np.float32)
ky = np.array([[-1,-1,-1],
               [ 0, 0, 0],
               [ 1, 1, 1]], dtype=np.float32)

gx = cv2.filter2D(img, cv2.CV_32F, kx)
gy = cv2.filter2D(img, cv2.CV_32F, ky)

mag = cv2.magnitude(gx, gy)
out = cv2.convertScaleAbs(mag)
show_result(out)
""",
        "Scharr": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

sx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
sy = cv2.Scharr(img, cv2.CV_32F, 0, 1)

mag = cv2.magnitude(sx, sy)
out = cv2.convertScaleAbs(mag)
show_result(out)
""",
        "Robert": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

kx = np.array([[1,0],
               [0,-1]], dtype=np.float32)
ky = np.array([[0,1],
               [-1,0]], dtype=np.float32)

gx = cv2.filter2D(img, cv2.CV_32F, kx)
gy = cv2.filter2D(img, cv2.CV_32F, ky)

mag = cv2.magnitude(gx, gy)
out = cv2.convertScaleAbs(mag)
show_result(out)
""",
        "Opening (Erosion → Dilation)": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
out = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
show_result(out)
""",
        "Closing (Dilation → Erosion)": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
out = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
show_result(out)
""",
        "Morph Gradient": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
out = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
show_result(out)
""",
        "Skeletonization": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

skel = np.zeros_like(bw)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

temp = bw.copy()
while True:
    eroded = cv2.erode(temp, kernel)
    opened = cv2.dilate(eroded, kernel)
    residue = cv2.subtract(temp, opened)
    skel = cv2.bitwise_or(skel, residue)
    temp = eroded
    if cv2.countNonZero(temp) == 0:
        break

show_result(skel)
""",
        "Distance transform": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
out = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show_result(out)
""",
        "Watershed": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
sure_fg = sure_fg.astype(np.uint8)
unknown = cv2.subtract(sure_bg, sure_fg)

_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)

out = img.copy()
out[markers == -1] = [0, 0, 255]
show_result(out)
""",
        "Connected Components": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

_, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

h, w = bw.shape
out = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(1, num):
    out[labels == i] = (0, 255, 0)

show_result(out)
""",
        "Grabcut": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape[:2]
rect = (w//8, h//8, w*3//4, h*3//4)

mask = np.zeros((h, w), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

fg = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)
out = img * fg[:, :, None]
show_result(out)
""",
        "Shi-Tomasi": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
out = img.copy()
if corners is not None:
    corners = np.int32(corners)
    for x, y in corners.reshape(-1, 2):
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)

show_result(out)
""",
        "FAST": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
kps = fast.detect(gray, None)

out = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_result(out)
""",
        "SURF": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

if not hasattr(cv2, "xfeatures2d"):
    raise RuntimeError("cv2.xfeatures2d가 없습니다. opencv-contrib-python 설치가 필요합니다.")

surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
kps, des = surf.detectAndCompute(img, None)

out = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_result(out)
""",
        "BRIEF": """import cv2

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

if not hasattr(cv2, "xfeatures2d"):
    raise RuntimeError("cv2.xfeatures2d가 없습니다. opencv-contrib-python 설치가 필요합니다.")

star = cv2.xfeatures2d.StarDetector_create()
kps = star.detect(img, None)

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
kps, des = brief.compute(img, kps)

out = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_result(out)
""",
        "Convex Hull": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

out = img.copy()
for cnt in contours:
    if cv2.contourArea(cnt) < 200:
        continue
    hull = cv2.convexHull(cnt)
    cv2.polylines(out, [hull], True, (0, 0, 255), 2)

show_result(out)
""",
        "Bounding Box": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

out = img.copy()
for cnt in contours:
    if cv2.contourArea(cnt) < 200:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.polylines(out, [box], True, (0, 0, 255), 2)

show_result(out)
""",
        "Translation": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape[:2]
dx, dy = 50, 30
M = np.float32([[1, 0, dx],
                [0, 1, dy]])
out = cv2.warpAffine(img, M, (w, h))
show_result(out)
""",
        "Rotation": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape[:2]
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, angle=25, scale=1.0)
out = cv2.warpAffine(img, M, (w, h))
show_result(out)
""",
        "Scale": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
show_result(out)
""",
        "Affine": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape[:2]
src = np.float32([[0,0], [w-1,0], [0,h-1]])
dst = np.float32([[20,30], [w-40,10], [40,h-30]])

M = cv2.getAffineTransform(src, dst)
out = cv2.warpAffine(img, M, (w, h))
show_result(out)
""",
        "Similarity": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape[:2]
center = (w//2, h//2)

M = cv2.getRotationMatrix2D(center, angle=20, scale=0.9)
M[0,2] += 30
M[1,2] += 10

out = cv2.warpAffine(img, M, (w, h))
show_result(out)
""",
        "Rigid (Euclidean)": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape[:2]
center = (w//2, w//2)

M = cv2.getRotationMatrix2D(center, angle=15, scale=1.0)
M[0,2] += 40
M[1,2] += -10

out = cv2.warpAffine(img, M, (w, h))
show_result(out)
""",
        "Nearest": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
show_result(out)
""",
        "Bilinear": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
show_result(out)
""",
        "Bicubic": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
show_result(out)
""",
        "Area": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

out = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
show_result(out)
""",
        "FFT": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
mag = np.log(np.abs(fshift) + 1.0)

out = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show_result(out)
""",
        "DCT": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

img_f = np.float32(img) / 255.0
dct = cv2.dct(img_f)

mag = np.log(np.abs(dct) + 1e-6)
out = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show_result(out)
""",
        "High-pass": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

mask = np.ones((h, w), np.uint8)
r = min(h, w) // 10
cy, cx = h//2, w//2
mask[cy-r:cy+r, cx-r:cx+r] = 0

fshift_hp = fshift * mask
img_back = np.fft.ifft2(np.fft.ifftshift(fshift_hp))
img_back = np.abs(img_back)

out = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show_result(out)
""",
        "Low-pass": """import cv2
import numpy as np

img = imread_unicode(r"{IMAGE_PATH}", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

h, w = img.shape
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

mask = np.zeros((h, w), np.uint8)
r = min(h, w) // 10
cy, cx = h//2, w//2
mask[cy-r:cy+r, cx-r:cx+r] = 1

fshift_lp = fshift * mask
img_back = np.fft.ifft2(np.fft.ifftshift(fshift_lp))
img_back = np.abs(img_back)

out = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show_result(out)
""",
        "Lab": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
show_result(lab)
""",
        "YCrCb": """import cv2

img = imread_unicode(r"{IMAGE_PATH}")
if img is None:
    raise RuntimeError("이미지를 읽을 수 없습니다. 경로를 확인하세요.")

ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
show_result(ycrcb)
""",
    })

    def build_code_for_selected_type(type_name: str) -> str:
        img_path = sanitize_path(CURRENT_IMAGE_PATH) if CURRENT_IMAGE_PATH else "image.jpg"
        template = snippet_by_type.get(type_name)
        if not template:
            return (
                "import cv2\n\n"
                f"# 선택 항목: {type_name}\n"
                "# 이 항목에 대한 샘플 코드는 아직 매핑되어 있지 않습니다.\n"
            )
        return template.replace("{IMAGE_PATH}", img_path)

    def on_tax_select(event):
        sel = tree_tax.selection()
        if not sel:
            return
        item = sel[0]
        values = tree_tax.item(item, "values")
        type_name = values[2]
        code = build_code_for_selected_type(type_name)
        txt_code.delete("1.0", tk.END)
        txt_code.insert(tk.END, code)

    tree_tax.bind("<<TreeviewSelect>>", on_tax_select)

    def run_snippet():
        print_cheatsheet_widget_sizes()
        code = txt_code.get("1.0", tk.END)

        # ✅ 실행 환경에 imread_unicode 제공
        env = {"cv2": cv2, "np": np, "show_result": show_result, "imread_unicode": imread_unicode}

        try:
            exec(code, env, env)
        except cv2.error as e:
            if "!_src.empty()" in str(e):
                messagebox.showerror("코드 실행 에러", "이미지 경로를 확인해 주세요.")
            else:
                messagebox.showerror("코드 실행 에러", str(e))
        except RuntimeError as e:
            if "이미지를 읽을 수 없습니다" in str(e):
                messagebox.showerror("코드 실행 에러", "이미지 경로를 확인해 주세요.")
            else:
                messagebox.showerror("코드 실행 에러", str(e))
        except Exception as e:
            messagebox.showerror("코드 실행 에러", str(e))

    btn_run.configure(command=run_snippet)

    # ---------- 우측 하단: 응용 조합표 + 등록 버튼 ----------
    br = ttk.LabelFrame(bottom_right, text="응용 방법 별 최적 필터 조합 정보")
    br.pack(fill="both", expand=True)

    top_bar = ttk.Frame(br)
    top_bar.pack(fill="x", padx=8, pady=6)

    ttk.Label(top_bar, text="응용 필터 조합(ini 관리)").pack(side="left")

    btn_add = ttk.Button(top_bar, text="등록")
    btn_add.pack(side="right")

    tree_scen = ttk.Treeview(br, columns=("scenario", "goal", "pipeline"), show="headings", height=14)
    tree_scen.heading("scenario", text="이미지 상황별 최적 필터 조합(현행 유지)")
    tree_scen.heading("goal", text="목표")
    tree_scen.heading("pipeline", text="권장 파이프라인")
    tree_scen.column("scenario", width=260, anchor="w")
    tree_scen.column("goal", width=240, anchor="w")
    tree_scen.column("pipeline", width=520, anchor="w")
    tree_scen.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def refresh_scen_tree():
        for r in tree_scen.get_children():
            tree_scen.delete(r)
        items = load_pipelines_from_ini()
        for sec, sc, goal, pipe in items:
            tree_scen.insert("", "end", iid=sec, values=(sc, goal, pipe))

    refresh_scen_tree()

    def open_add_dialog():
        dlg = tk.Toplevel(parent)
        dlg.title("필터 조합 정보 등록")
        dlg.geometry("600x260")
        dlg.grab_set()

        frm = ttk.Frame(dlg)
        frm.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Label(frm, text="이미지 상황").grid(row=0, column=0, sticky="w", pady=4)
        e1 = ttk.Entry(frm)
        e1.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="목표").grid(row=1, column=0, sticky="w", pady=4)
        e2 = ttk.Entry(frm)
        e2.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(frm, text="권장 파이프라인").grid(row=2, column=0, sticky="w", pady=4)
        e3 = ttk.Entry(frm)
        e3.grid(row=2, column=1, sticky="ew", pady=4)

        frm.columnconfigure(1, weight=1)

        def on_save():
            sc = e1.get().strip()
            goal = e2.get().strip()
            pipe = e3.get().strip()

            if not sc or not goal or not pipe:
                messagebox.showwarning("입력 필요", "모든 항목(상황/목표/파이프라인)을 입력해 주세요.")
                return

            add_pipeline_to_ini(sc, goal, pipe)
            refresh_scen_tree()
            dlg.destroy()

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, sticky="e", pady=10)
        ttk.Button(btns, text="저장", command=on_save).pack(side="left", padx=5)
        ttk.Button(btns, text="취소", command=dlg.destroy).pack(side="left")

    btn_add.configure(command=open_add_dialog)


def release_single_instance_lock(fd):
    """
    단일 실행 락 해제
    """
    try:
        if fd:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)

            fd.close()
    except Exception:
        pass

# -------------------------------
# 9) 메인: 탭 구성 + 시작 시 최대화
# -------------------------------
def main():
    # lock_fd = acquire_single_instance_lock()
    # if lock_fd is None:
    #     root = tk.Tk()
    #     root.withdraw()
    #     # messagebox.showwarning(
    #     #     title="실행 제한",
    #     #     message="이미 프로그램이 실행 중입니다.\n(중복 실행 방지)"
    #     # )
    #     root.destroy()   # ★ 반드시 Tk 인스턴스 정리
    #     return

    try:
        root = tk.Tk()
        root.title("이미지 자동 필터 추천 + 치트시트")
        root.state("zoomed")

        notebook = ttk.Notebook(root)
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)

        notebook.add(tab1, text="자동 분석")
        notebook.add(tab2, text="치트시트 / 샘플코드")
        notebook.pack(fill="both", expand=True)

        AutoFilterApp(tab1)
        build_cheatsheet_tab(tab2)

        root.mainloop()

    finally:
        
        # release_single_instance_lock(lock_fd)
        # 정상 종료 시 락 해제
        try:
            # os.close(lock_fd)
            os.remove(os.path.join(tempfile.gettempdir(), "opencv_cheatsheet_app.lock"))
        except Exception:
            pass


if __name__ == "__main__":
    main()
