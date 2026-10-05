"""Script tự động đọc kết quả output từ các file .ipynb và xuất ra ảnh PNG đẹp mắt
để nộp bài vào submission/screenshots/ theo đúng chuẩn rubric.

Cách dùng:
    python scripts/export_screenshots.py              # Xuất toàn bộ các notebook đã chạy
    python scripts/export_screenshots.py --nb 1       # Chỉ xuất riêng NB1
    python scripts/export_screenshots.py --nb 2       # Chỉ xuất riêng NB2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Thử import PIL (Pillow), nếu chưa có thì hướng dẫn hoặc tự cài
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("[!] Thư viện Pillow chưa được cài đặt.")
    print("[*] Đang cài đặt Pillow để vẽ ảnh: pip install pillow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT / "notebooks"
SCREENSHOTS_DIR = ROOT / "submission" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình giao diện Dark Mode (VS Code style)
BG_COLOR = (30, 30, 30)         # #1e1e1e
CARD_BG = (37, 37, 38)          # #252526
HEADER_BG = (45, 45, 48)        # #2d2d30
TEXT_MAIN = (212, 212, 212)     # #d4d4d4
TEXT_COMMENT = (106, 153, 85)   # #6a9955 (xanh lá comment)
TEXT_PROMPT = (0, 122, 204)     # #007acc (xanh dương In [x])
OUTPUT_BG = (24, 24, 24)        # #181818
OUTPUT_TEXT = (220, 220, 220)   # #dcdcdc
SUCCESS_TEXT = (78, 201, 176)   # #4ec9b0 (xanh ngọc)
BORDER_COLOR = (60, 60, 60)     # #3c3c3c

# Các cell cần chụp cho từng notebook theo chuẩn Rubric
TARGET_CELLS = {
    1: {
        "title": "NB1: Embeddings & Vector Indexing (Qdrant)",
        "file": "01_embeddings_index.ipynb",
        # Các từ khóa nhận diện cell cần chụp
        "match_keywords": ["Indexed:", "Query: 'cloud computing", "Query (paraphrase):"],
        "out_file": "nb1.png",
    },
    2: {
        "title": "NB2: Hybrid Search & RRF Precision@10",
        "file": "02_hybrid_search_rrf.ipynb",
        "match_keywords": ["Precision@10", "Avg Precision", "mixed", "paraphrase"],
        "out_file": "nb2.png",
    },
    3: {
        "title": "NB3: Search API Latency Benchmark",
        "file": "03_search_api_benchmark.ipynb",
        "match_keywords": ["latency_ms", "P50", "P95", "P99", "FastAPI"],
        "out_file": "nb3.png",
    },
    4: {
        "title": "NB4: Feast Feature Store",
        "file": "04_feast_feature_store.ipynb",
        "match_keywords": ["feast apply", "materialize", "get_online_features", "get_historical_features"],
        "out_file": "nb4.png",
    },
    5: {
        "title": "NB5: Filtered Search (Recall Cliff & Over-Fetch)",
        "file": "05_filtered_search.ipynb",
        "match_keywords": ["sel%", "fANN", "post", "ladder", "over-fetch"],
        "out_file": "nb5.png",
    },
    6: {
        "title": "NB6: Agentic Retrieval (Decomposition & Context)",
        "file": "06_agent_retrieval.ipynb",
        "match_keywords": ["ngân sách", "single-shot", "agentic", "build_context"],
        "out_file": "nb6.png",
    },
    7: {
        "title": "NB7: Semantic Cache (Threshold Sweep & Multi-tenant)",
        "file": "07_semantic_cache.ipynb",
        "match_keywords": ["sweep", "tiết kiệm", "rò chéo", "leak", "MISS"],
        "out_file": "nb7.png",
    },
    8: {
        "title": "NB8: Feature Engineering (Leakage & PIT Join)",
        "file": "08_feature_engineering.ipynb",
        "match_keywords": ["leakage", "target-naive", "PIT", "on-demand", "gap"],
        "out_file": "nb8.png",
    },
}


def load_font(size: int = 14) -> ImageFont.ImageFont:
    """Tải font mono tiêu chuẩn, fallback về font mặc định nếu không tìm thấy."""
    font_candidates = [
        "consola.ttf", "consolas.ttf", "cascadiacode.ttf", "cour.ttf",
        "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"
    ]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def extract_evidence_cells(nb_path: Path, keywords: list[str]) -> list[dict]:
    """Đọc file notebook và lọc ra các cell chứa output khớp với tiêu chí rubric."""
    if not nb_path.exists():
        return []

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    selected = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs", [])
        if not outputs:
            continue

        # Gom toàn bộ output text của cell
        out_text = ""
        for out in outputs:
            if "text" in out:
                if isinstance(out["text"], list):
                    out_text += "".join(out["text"])
                else:
                    out_text += str(out["text"])
            elif "data" in out and "text/plain" in out["data"]:
                data = out["data"]["text/plain"]
                out_text += "".join(data) if isinstance(data, list) else str(data)

        # Kiểm tra xem cell này có chứa kết quả cần chụp không
        matches = any(k.lower() in out_text.lower() for k in keywords)
        # Hoặc kiểm tra trong source code
        source_text = "".join(cell.get("source", []))
        if matches or any(k.lower() in source_text.lower() for k in keywords):
            selected.append({
                "exec_count": cell.get("execution_count", "?"),
                "source": source_text.strip(),
                "output": out_text.strip(),
            })

    # Nếu không khớp keyword nào nhưng có cell có output, lấy 3 cell code cuối cùng
    if not selected:
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code" and cell.get("outputs"):
                out_text = ""
                for out in cell["outputs"]:
                    if "text" in out:
                        out_text += "".join(out["text"]) if isinstance(out["text"], list) else str(out["text"])
                    elif "data" in out and "text/plain" in out["data"]:
                        d = out["data"]["text/plain"]
                        out_text += "".join(d) if isinstance(d, list) else str(d)
                selected.append({
                    "exec_count": cell.get("execution_count", "?"),
                    "source": "".join(cell.get("source", [])).strip(),
                    "output": out_text.strip(),
                })
        selected = selected[-3:]

    return selected


def render_notebook_image(title: str, cells_data: list[dict], out_path: Path) -> None:
    """Vẽ ảnh PNG giao diện Dark Mode hiển thị code và output của các cell."""
    font = load_font(15)
    font_bold = load_font(16)
    font_title = load_font(18)

    # 1. Tính toán trước chiều cao của ảnh
    line_h = 22
    width = 1100
    y = 70  # Khoảng trống sau header

    rendered_blocks = []
    for cell in cells_data:
        # Code block
        code_lines = cell["source"].split("\n")
        # Giới hạn số dòng code hiển thị để ảnh không quá dài (lấy 10 dòng đầu/cuối quan trọng)
        if len(code_lines) > 16:
            code_lines = code_lines[:8] + ["    # ... (đã rút gọn) ..."] + code_lines[-5:]

        # Output block
        out_lines = cell["output"].split("\n")
        if len(out_lines) > 20:
            out_lines = out_lines[:15] + ["... (còn tiếp) ..."]

        block_height = (len(code_lines) * line_h + 30) + (len(out_lines) * line_h + 30) + 40
        rendered_blocks.append({
            "exec_count": cell["exec_count"],
            "code_lines": code_lines,
            "out_lines": out_lines,
            "y": y,
            "height": block_height,
        })
        y += block_height

    height = max(y + 30, 400)

    # 2. Tạo canvas ảnh
    img = Image.new("RGB", (width, height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 3. Vẽ Top Header
    draw.rectangle([(0, 0), (width, 50)], fill=HEADER_BG)
    draw.line([(0, 50), (width, 50)], fill=BORDER_COLOR, width=2)
    # 3 nút cửa sổ (Mac/VS Code style)
    draw.ellipse([(15, 18), (27, 30)], fill=(255, 95, 86))
    draw.ellipse([(35, 18), (47, 30)], fill=(255, 189, 46))
    draw.ellipse([(55, 18), (67, 30)], fill=(39, 201, 63))
    # Tiêu đề
    draw.text((80, 15), title, fill=TEXT_MAIN, font=font_title)

    # 4. Vẽ từng cell
    for block in rendered_blocks:
        cur_y = block["y"]
        exec_count = block["exec_count"]

        # Card container cho 1 cell
        draw.rectangle(
            [(25, cur_y), (width - 25, cur_y + block["height"] - 20)],
            fill=CARD_BG,
            outline=BORDER_COLOR,
            width=1,
        )

        # Prompt In [x]:
        prompt_str = f"In [{exec_count}]:"
        draw.text((40, cur_y + 15), prompt_str, fill=TEXT_PROMPT, font=font_bold)

        # Khối Code Input
        code_y = cur_y + 15
        for line in block["code_lines"]:
            color = TEXT_COMMENT if line.strip().startswith("#") else TEXT_MAIN
            draw.text((120, code_y), line, fill=color, font=font)
            code_y += line_h

        # Khối Output
        out_start_y = code_y + 15
        out_box_h = len(block["out_lines"]) * line_h + 16
        draw.rectangle(
            [(40, out_start_y), (width - 40, out_start_y + out_box_h)],
            fill=OUTPUT_BG,
            outline=BORDER_COLOR,
            width=1,
        )

        # Vẽ từng dòng output
        line_y = out_start_y + 8
        for line in block["out_lines"]:
            # Đổi màu xanh cho các dòng thành công hoặc điểm cao
            c = SUCCESS_TEXT if any(w in line for w in ["Indexed:", "1000", "score=0.8", "True", "OK"]) else OUTPUT_TEXT
            draw.text((55, line_y), line, fill=c, font=font)
            line_y += line_h

    # 5. Lưu ảnh
    img.save(out_path, format="PNG")
    print(f"  [✓] Đã tạo ảnh thành công: {out_path.relative_to(ROOT)}")


def export_bonus_demo_screenshot() -> None:
    """Tự động chạy và chụp lại màn hình console của Bonus Challenge Demo."""
    demo_script = ROOT / "bonus" / "demo.py"
    if not demo_script.exists():
        return

    import contextlib
    import io

    # Chạy in-process với redirect_stdout: đảm bảo 100% tiếng Việt UTF-8 không bị lỗi Windows pipe
    buf = io.StringIO()
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from bonus.demo import main as run_demo
        with contextlib.redirect_stdout(buf):
            run_demo()
        output_text = buf.getvalue()
    except Exception as e:
        output_text = f"Lỗi chạy demo: {e}"

    if not output_text.strip():
        output_text = "[Bonus Demo] Không có output."

    out_lines = output_text.strip().split("\n")
    font = load_font(14)
    font_bold = load_font(15)
    font_title = load_font(18)

    line_h = 20
    width = 1100
    height = max(len(out_lines) * line_h + 100, 500)

    img = Image.new("RGB", (width, height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Top Header
    draw.rectangle([(0, 0), (width, 50)], fill=HEADER_BG)
    draw.line([(0, 50), (width, 50)], fill=BORDER_COLOR, width=2)
    draw.ellipse([(15, 18), (27, 30)], fill=(255, 95, 86))
    draw.ellipse([(35, 18), (47, 30)], fill=(255, 189, 46))
    draw.ellipse([(55, 18), (67, 30)], fill=(39, 201, 63))
    draw.text((80, 15), "Bonus Challenge: Hybrid Memory AI Agent (demo.py)", fill=TEXT_MAIN, font=font_title)

    # Khung Terminal
    term_y = 65
    draw.rectangle(
        [(25, term_y), (width - 25, height - 25)],
        fill=OUTPUT_BG,
        outline=BORDER_COLOR,
        width=1,
    )

    y = term_y + 15
    for line in out_lines:
        c = OUTPUT_TEXT
        if "=== [ASSEMBLED" in line or "===" in line:
            c = TEXT_PROMPT
        elif "DEMO HOÀN TẤT" in line or "Exit code: 0" in line or "Đã nạp thành công" in line:
            c = SUCCESS_TEXT
        elif "❓ Câu hỏi" in line or "💡 Mục tiêu" in line:
            c = (255, 215, 0)  # Gold
        elif line.strip().startswith("- [0."):
            c = (156, 220, 254)  # Light cyan

        draw.text((45, y), line, fill=c, font=font)
        y += line_h

    out_path = SCREENSHOTS_DIR / "bonus_demo.png"
    img.save(out_path, format="PNG")
    print(f"  [✓] Đã tạo ảnh thành công: {out_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Chụp output notebook sang PNG")
    parser.add_argument("--nb", type=int, choices=list(range(1, 9)), help="Số thứ tự notebook (1 đến 8)")
    args = parser.parse_args()

    targets = [args.nb] if args.nb else list(range(1, 9))
    print(f"[*] Bắt đầu xuất ảnh nộp bài vào: {SCREENSHOTS_DIR.relative_to(ROOT)}/")

    exported_count = 0
    for nb_id in targets:
        info = TARGET_CELLS[nb_id]
        nb_path = NOTEBOOKS_DIR / info["file"]
        out_file = SCREENSHOTS_DIR / info["out_file"]

        if not nb_path.exists():
            print(f"  [-] Chưa tìm thấy {nb_path.name}, bỏ qua.")
            continue

        cells_data = extract_evidence_cells(nb_path, info["match_keywords"])
        if not cells_data:
            print(f"  [-] {nb_path.name} chưa có output đã chạy, bỏ qua.")
            continue

        render_notebook_image(info["title"], cells_data, out_file)
        exported_count += 1

    # Nếu có thư mục bonus, xuất thêm ảnh bonus_demo.png
    if (ROOT / "bonus" / "demo.py").exists() and not args.nb:
        export_bonus_demo_screenshot()
        exported_count += 1

    print(f"[Done] Đã hoàn tất xuất {exported_count} ảnh chụp màn hình!")


if __name__ == "__main__":
    main()
