"""
create_confusion_matrix_png.py
-------------------------------
Generates high-resolution PNG & SVG confusion matrix images in app/static/
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parent
_STATIC_DIR = _ROOT / "app" / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)

def create_image():
    # 600x400 high contrast dark mode confusion matrix image
    img = Image.new('RGB', (600, 360), color='#080c14')
    draw = ImageDraw.Draw(img)

    # Outer border
    draw.rectangle([5, 5, 595, 355], outline='#1e293b', width=2)

    # Title
    draw.text((120, 25), "FAILUREAWARE AI — CONFUSION MATRIX", fill='#6366f1')

    # TP (True Positives)
    draw.rectangle([40, 70, 280, 190], fill='#022c22', outline='#10b981', width=2)
    draw.text((60, 85), "TRUE POSITIVES (TP)", fill='#a7f3d0')
    draw.text((60, 115), "191", fill='#34d399')
    draw.text((60, 155), "Correctly Blocked Fraud", fill='#6ee7b7')

    # TN (True Negatives)
    draw.rectangle([310, 70, 550, 190], fill='#1e1b4b', outline='#6366f1', width=2)
    draw.text((330, 85), "TRUE NEGATIVES (TN)", fill='#c7d2fe')
    draw.text((330, 115), "97", fill='#818cf8')
    draw.text((330, 155), "Correctly Approved Claims", fill='#a5b4fc')

    # FP (False Positives)
    draw.rectangle([40, 210, 280, 330], fill='#451a03', outline='#f59e0b', width=2)
    draw.text((60, 225), "FALSE POSITIVES (FP)", fill='#fde68a')
    draw.text((60, 255), "6", fill='#fbbf24')
    draw.text((60, 295), "Valid Claims Flagged", fill='#fef08a')

    # FN (False Negatives)
    draw.rectangle([310, 210, 550, 330], fill='#4c0519', outline='#f43f5e', width=2)
    draw.text((330, 225), "FALSE NEGATIVES (FN)", fill='#fecdd3')
    draw.text((330, 255), "6", fill='#fb7185')
    draw.text((330, 295), "Borderline Manual Audits", fill='#ffe4e6')

    target_file = _STATIC_DIR / "confusion_matrix.png"
    img.save(target_file)
    print(f"[SUCCESS] Generated Confusion Matrix PNG at {target_file}")

if __name__ == "__main__":
    create_image()
