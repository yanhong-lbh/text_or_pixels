import os
import subprocess
import math
import tempfile
from typing import Tuple
from PIL import Image as PIL_Image
from pdf2image import convert_from_path
import pandas as pd
import sys
import json
import argparse
from datetime import datetime

def escape_latex_special_chars(text_input):
    """Escapes LaTeX special characters in a string."""
    escape_map = {
        # Must be first if other replacements could introduce backslashes
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    # Build a regex for efficiency if text can be very long,
    # but for typical short texts, simple replace is fine.
    # Order of keys in dict is not guaranteed for Python < 3.7,
    # so for \ to be first, ensure it's handled separately or use OrderedDict.
    # For this set of non-overlapping simple characters (except \), it's usually okay.
    # To be safe, handle backslash first:
    if "\\" in escape_map: # Check if backslash needs escaping
        text_input = text_input.replace("\\", escape_map["\\"])
    
    for char, replacement_latex in escape_map.items():
        if char == "\\": # Already handled
            continue
        text_input = text_input.replace(char, replacement_latex)
    return text_input



def text_to_image(
    text,
    width=600,          
    height=640,        
    margin_px=10,
    font_path=None,     
    target_fill_ratio=0.85,  
    dpi=300             
):


    typographic_replacements = {
        "'": "'",  # U+2019 RIGHT SINGLE QUOTATION MARK -> APOSTROPHE
        "'": "'",  # U+2018 LEFT SINGLE QUOTATION MARK -> APOSTROPHE
        """: '"',  # U+201C LEFT DOUBLE QUOTATION MARK -> QUOTATION MARK
        """: '"',  # U+201D RIGHT DOUBLE QUOTATION MARK -> QUOTATION MARK
        "–": "-",  # U+2013 EN DASH -> HYPHEN-MINUS
        "—": "--", # U+2014 EM DASH -> two HYPHEN-MINUS (LaTeX makes this an em-dash)
        "…": "...", # U+2026 HORIZONTAL ELLIPSIS -> three periods
    }
    for original, replacement in typographic_replacements.items():
        text = text.replace(original, replacement)

    # 2. Escape LaTeX special characters
    text = escape_latex_special_chars(text)

    def calculate_bbox_fill_ratio(img, bg_threshold=240):
        """Calculate what percentage of image area is filled with content."""
        gray = img.convert("L")
        mask = gray.point(lambda p: 0 if p > bg_threshold else 255, mode="1")
        bbox = mask.getbbox()
        if bbox is None:
            return 0.0

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_area = bbox_width * bbox_height
        total_area = gray.width * gray.height

        return bbox_area / total_area

    def test_font_size_and_render(font_size):
        """Test a font size and return the rendered image and fill ratio."""
        # Convert pixels to points for LaTeX
        #width_pt = width * 72 / dpi
        width_pt = 2*width
        height_pt = 2*height
        #height_pt = height * 72 / dpi

        tex_content = f"""
\\documentclass{{article}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{helvet}}
\\usepackage[paperwidth={width_pt}pt,paperheight={height_pt}pt,margin={margin_px}pt]{{geometry}}
\\renewcommand{{\\familydefault}}{{\\sfdefault}}
\\pagestyle{{empty}}
\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{0pt}}
\\linespread{{1.2}}

\\begin{{document}}
\\fontsize{{{font_size}pt}}{{{font_size * 1.1}pt}}\\selectfont
\\raggedright
{text}
\\end{{document}}
"""

        #temp_dir = "temp_latex_files"
        temp_dir = f"temp_latex_files_{os.getpid()}"
        os.makedirs(temp_dir, exist_ok=True)
        tex_file_path = os.path.join(temp_dir, "temp_doc.tex")
        pdf_file_path = os.path.join(temp_dir, "temp_doc.pdf")

        with open(tex_file_path, "w", encoding="utf-8") as f:
            f.write(tex_content)

        try:
            process = subprocess.run(
                ["tectonic", "temp_doc.tex"],
                check=True,
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            if not os.path.exists(pdf_file_path):
                return None, 0.0

            pages = convert_from_path(pdf_file_path, dpi=dpi)
            if not pages or len(pages) != 1:  
                return None, 0.0

            page_img = pages[0].convert("RGB")
            
            page_img = page_img.resize((width, height), PIL_Image.Resampling.LANCZOS)

            fill_ratio = calculate_bbox_fill_ratio(page_img)
            return page_img, fill_ratio

        except:
            return None, 0.0
        finally:

            try:
                if os.path.exists(tex_file_path): os.remove(tex_file_path)
                if os.path.exists(pdf_file_path): os.remove(pdf_file_path)
                for ext in ['.aux', '.log', '.synctex.gz', '.fls', '.fdb_latexmk']:
                    aux_file = os.path.join(temp_dir, f"temp_doc{ext}")
                    if os.path.exists(aux_file):
                        os.remove(aux_file)
            except:
                pass


    font_candidates = [150, 120, 100, 80, 60, 50, 45, 40, 35, 30, 25]
    for size in range(24, 0, -1):
        font_candidates.extend([size, size - 0.5])
    font_candidates.append(0.5)


    font_candidates = sorted(list(set(font_candidates)), reverse=True)

    best_font = None
    best_image = None
    best_ratio = 0.0

    for font_size in font_candidates:
        page_img, fill_ratio = test_font_size_and_render(font_size)

        if page_img is None:
            continue


        if best_font is None or font_size > best_font:
            best_font = font_size
            best_image = page_img.copy()
            best_ratio = fill_ratio

        if fill_ratio >= target_fill_ratio:
            break


    #temp_dir = "temp_latex_files"
    temp_dir = f"temp_latex_files_{os.getpid()}"
    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass

    return best_image


def generate_images_only(df, images_dir, start_idx=0, end_idx=None, width=600, height=640, dpi=300):
    os.makedirs(images_dir, exist_ok=True)
    
    if end_idx is None:
        df_slice = df.iloc[start_idx:]
    else:
        df_slice = df.iloc[start_idx:end_idx]
    
    total_rows = len(df_slice)
    print(f"Task processing {total_rows} rows (index {start_idx} to {end_idx})")
    
    for idx, (_, row) in enumerate(df_slice.iterrows()):
        try:
            doc_dict = row['doc']
            text_content = doc_dict['input']
            #text_content = row['chapter']
            
            image = text_to_image(text_content, width=width, height=height, dpi=dpi)
            
            image_filename = f"input_{row['doc_id']}.png"
            image_path = os.path.join(images_dir, image_filename)
            image.save(image_path)
            
            print(f"Generated: {image_filename} ({idx+1}/{total_rows})")
            
        except Exception as e:
            print(f"Failed {row['doc_id']}: {e}")



def main():
    parser = argparse.ArgumentParser(description='text_to_image')
    
    parser.add_argument('--input', '-i', required=True, help='Path to JSONL file')
    parser.add_argument('--data_dir', '-d', required=True, help='Directory to save images')
    parser.add_argument('--width', type=int, default=800, help='Image width')
    parser.add_argument('--height', type=int, default=600, help='Image height')
    parser.add_argument('--dpi', type=int, default=300, help='DPI')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=None, help='End index')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
            print(f"File does not exist: {args.input}")
            return 1
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)
    
    try:
        df = pd.read_json(args.input, lines=True)
        print(f"Read {len(df)} records")
        
        generate_images_only(
            df,
            args.data_dir,  # Use data_dir directly, no subfolder
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            width=args.width,
            height=args.height,
            dpi=args.dpi
        )
        
        print("Complete!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())