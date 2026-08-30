# ============================================================
# HUGGING FACE CPU OPTIMIZED PADDLEOCR FASTAPI
# ============================================================

# IMPORTANT:
# These MUST be set BEFORE importing paddle / paddleocr.
# ============================================================

import os

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

# Hugging Face Spaces uses port 7860
PORT = int(os.environ.get("PORT", "7860"))

# ============================================================
# IMPORTS
# ============================================================

import re
import uuid
import shutil
import json
import tempfile

from typing import Dict, Any, List, Optional

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form
)

from fastapi.responses import JSONResponse

from paddleocr import PaddleOCR


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Dynamic Image Key Value OCR API",
    description="Extract only requested key-value pairs from image",
    version="5.0.0"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Use /tmp on Hugging Face.
# Do not depend on persistent local storage.

UPLOAD_DIR = os.environ.get(
    "UPLOAD_DIR",
    "/tmp/ocr_uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# Maximum upload size:
# 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024


# ============================================================
# OCR INITIALIZATION
#
# IMPORTANT:
# This happens ONCE when the application starts.
#
# Do NOT create PaddleOCR inside /extract.
# ============================================================

print("==============================================")
print("Loading PaddleOCR...")
print("CPU MODE")
print("MKLDNN DISABLED")
print("==============================================")


try:

    ocr = PaddleOCR(
        lang="en",

        # CPU friendly settings
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    print("==============================================")
    print("PaddleOCR loaded successfully.")
    print("==============================================")


except Exception as e:

    print("==============================================")
    print("PADDLE OCR INITIALIZATION ERROR")
    print("==============================================")

    print(str(e))

    print("==============================================")

    raise


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "Dynamic Image OCR API is running",
        "endpoint": "POST /extract",
        "platform": "Hugging Face Spaces",
        "ocr": "PaddleOCR CPU"
    }


@app.get("/health")
def health():

    return {
        "status": "UP",
        "ocr_loaded": ocr is not None
    }


# ============================================================
# CLEAN KEY
# ============================================================

def clean_key(
        key: str
) -> str:

    if not key:
        return ""

    key = key.strip()

    key = re.sub(
        r"^[\-\:\=\.\,\s]+",
        "",
        key
    )

    key = re.sub(
        r"[\:\=\-\.\,\s]+$",
        "",
        key
    )

    return key.strip()


# ============================================================
# CLEAN VALUE
# ============================================================

def clean_value(
        value: str
) -> str:

    if not value:
        return ""

    value = value.strip()

    value = re.sub(
        r"^[\:\=\s]+",
        "",
        value
    )

    return value.strip()


# ============================================================
# CLEAN NUMERIC VALUE
# ============================================================

def clean_numeric_value(
        value: str
) -> str:

    if not value:
        return ""

    value = value.strip()

    value = re.sub(
        r"^[\s:=]+",
        "",
        value
    )

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        value
    )

    if match:
        return match.group(0)

    return ""


# ============================================================
# CHECK NUMERIC VALUE
# ============================================================

def is_numeric_value(
        text: str
) -> bool:

    if not text:
        return False

    return bool(
        re.search(
            r"\d",
            text
        )
    )


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_key(
        text: str
) -> str:

    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"\s+",
        "",
        text
    )

    text = text.upper()

    text = re.sub(
        r"^[\-\:\=\.\,\s]+",
        "",
        text
    )

    text = re.sub(
        r"[\:\=\-\.\,\s]+$",
        "",
        text
    )

    return text


# ============================================================
# NORMALIZE REQUESTED KEYS
# ============================================================

def normalize_requested_keys(
        keys: List[str]
) -> List[str]:

    normalized = []

    for key in keys:

        if not key:
            continue

        key = clean_key(key)

        if not key:
            continue

        if key not in normalized:
            normalized.append(key)

    return normalized


# ============================================================
# COMPARE KEYS
# ============================================================

def keys_match(
        ocr_key: str,
        requested_key: str
) -> bool:

    if not ocr_key or not requested_key:
        return False

    a = normalize_key(
        ocr_key
    )

    b = normalize_key(
        requested_key
    )

    return a == b


# ============================================================
# FIND REQUESTED KEY
# ============================================================

def find_requested_key(
        line: str,
        requested_keys: List[str]
) -> Optional[str]:

    if not line:
        return None

    line_clean = line.strip()

    # ========================================================
    # DIRECT KEY MATCH
    # ========================================================

    for requested_key in requested_keys:

        if keys_match(
                line_clean,
                requested_key
        ):
            return requested_key

    # ========================================================
    # KEY WITH ":" OR "="
    # ========================================================

    separator_match = re.match(
        r"^\s*(.*?)\s*[:=]\s*(.*)$",
        line_clean
    )

    if separator_match:

        key_part = separator_match.group(1).strip()

        for requested_key in requested_keys:

            if keys_match(
                    key_part,
                    requested_key
            ):
                return requested_key

    # ========================================================
    # KEY FOLLOWED BY VALUE
    # ========================================================

    for requested_key in requested_keys:

        key_normalized = normalize_key(
            requested_key
        )

        line_normalized = normalize_key(
            line_clean
        )

        if line_normalized.startswith(
                key_normalized
        ):

            remaining = line_normalized[
                len(key_normalized):
            ]

            if (
                    not remaining
                    or remaining[0].isdigit()
                    or remaining[0] in ":=-"
            ):
                return requested_key

    return None


# ============================================================
# EXTRACT VALUE FROM SAME LINE
# ============================================================

def extract_value_from_same_line(
        line: str,
        requested_key: str
) -> str:

    if not line:
        return ""

    line = line.strip()

    # ========================================================
    # Key:Value
    # ========================================================

    match = re.match(
        r"^\s*(.*?)\s*:\s*(.*?)\s*$",
        line
    )

    if match:

        key_part = match.group(1).strip()

        value_part = match.group(2).strip()

        if keys_match(
                key_part,
                requested_key
        ):

            return clean_value(
                value_part
            )

    # ========================================================
    # Key=Value
    # ========================================================

    match = re.match(
        r"^\s*(.*?)\s*=\s*(.*?)\s*$",
        line
    )

    if match:

        key_part = match.group(1).strip()

        value_part = match.group(2).strip()

        if keys_match(
                key_part,
                requested_key
        ):

            return clean_value(
                value_part
            )

    # ========================================================
    # Key Value
    # ========================================================

    key_normalized = normalize_key(
        requested_key
    )

    line_normalized = normalize_key(
        line
    )

    if line_normalized.startswith(
            key_normalized
    ):

        key_length = len(
            requested_key
        )

        remaining = line[
            key_length:
        ].strip()

        remaining = re.sub(
            r"^[\:\=\-\s]+",
            "",
            remaining
        )

        if remaining:

            return clean_value(
                remaining
            )

    return ""


# ============================================================
# EXTRACT VALUE FROM FOLLOWING LINES
# ============================================================

def extract_value_from_next_lines(
        lines: List[str],
        current_index: int,
        requested_key: str,
        requested_keys: List[str]
) -> str:

    for j in range(
            current_index + 1,
            min(
                current_index + 6,
                len(lines)
            )
    ):

        next_line = lines[j].strip()

        if not next_line:
            continue

        # ====================================================
        # Stop if another requested key appears.
        # ====================================================

        another_key = find_requested_key(
            next_line,
            requested_keys
        )

        if another_key:
            break

        # ====================================================
        # SAME LINE VALUE
        # ====================================================

        same_line_value = extract_value_from_same_line(
            next_line,
            requested_key
        )

        if same_line_value:

            return same_line_value

        # ====================================================
        # NUMERIC VALUE
        # ====================================================

        if is_numeric_value(
                next_line
        ):

            numeric_value = clean_numeric_value(
                next_line
            )

            if numeric_value:

                return numeric_value

        # ====================================================
        # TEXT VALUE
        # ====================================================

        if len(
                next_line.strip()
        ) > 1:

            return clean_value(
                next_line
            )

    return ""


# ============================================================
# MAIN DYNAMIC EXTRACTION
# ============================================================

def extract_requested_key_values(
        lines: List[str],
        requested_keys: List[str]
) -> Dict[str, Any]:

    result = {}

    for i, line in enumerate(lines):

        line = line.strip()

        if not line:
            continue

        requested_key = find_requested_key(
            line,
            requested_keys
        )

        if not requested_key:
            continue

        # ====================================================
        # SAME LINE
        # ====================================================

        value = extract_value_from_same_line(
            line,
            requested_key
        )

        if value:

            if requested_key not in result:

                result[requested_key] = value

            continue

        # ====================================================
        # NEXT LINES
        # ====================================================

        value = extract_value_from_next_lines(
            lines,
            i,
            requested_key,
            requested_keys
        )

        if value:

            if requested_key not in result:

                result[requested_key] = value

    # ========================================================
    # PRESERVE REQUESTED KEY ORDER
    # ========================================================

    ordered_result = {}

    for requested_key in requested_keys:

        if requested_key in result:

            ordered_result[requested_key] = result[
                requested_key
            ]

    return ordered_result


# ============================================================
# OCR TEXT EXTRACTION
# ============================================================

def extract_text_from_image(
        image_path: str
) -> List[str]:

    all_text = []

    print("Running OCR...")

    try:

        result = ocr.predict(
            image_path
        )

        print("OCR completed.")

        # ====================================================
        # PROCESS OCR RESULT
        # ====================================================

        for res in result:

            try:

                data = None

                # ------------------------------------------------
                # res.json
                # ------------------------------------------------

                try:

                    data = res.json

                except Exception:

                    data = None

                # ------------------------------------------------
                # Callable JSON
                # ------------------------------------------------

                if callable(data):

                    data = data()

                # ------------------------------------------------
                # JSON string
                # ------------------------------------------------

                if isinstance(
                        data,
                        str
                ):

                    try:

                        data = json.loads(
                            data
                        )

                    except Exception:

                        data = None

                # ------------------------------------------------
                # Dictionary
                # ------------------------------------------------

                if isinstance(
                        data,
                        dict
                ):

                    result_data = data.get(
                        "res",
                        data
                    )

                    texts = result_data.get(
                        "rec_texts",
                        []
                    )

                    for text in texts:

                        if text is None:
                            continue

                        text = str(
                            text
                        ).strip()

                        if text:

                            all_text.append(
                                text
                            )

                # ------------------------------------------------
                # FALLBACK
                # ------------------------------------------------

                if not data:

                    try:

                        data = res["res"]

                        texts = data.get(
                            "rec_texts",
                            []
                        )

                        for text in texts:

                            if text is None:
                                continue

                            text = str(
                                text
                            ).strip()

                            if text:

                                all_text.append(
                                    text
                                )

                    except Exception:

                        pass

            except Exception as e:

                print(
                    "OCR result parsing error:",
                    str(e)
                )

    except Exception as e:

        print(
            "OCR execution error:",
            str(e)
        )

        raise

    # ========================================================
    # CLEAN OCR LINES
    # ========================================================

    cleaned_text = []

    for text in all_text:

        text = text.strip()

        if not text:
            continue

        cleaned_text.append(
            text
        )

    # ========================================================
    # PRINT OCR RESULT
    # ========================================================

    print("==============================================")
    print("Detected OCR lines")
    print("==============================================")

    for line in cleaned_text:

        print(
            " ->",
            line
        )

    print("==============================================")

    return cleaned_text


# ============================================================
# PARSE KEYS FROM POSTMAN
# ============================================================

def parse_keys(
        keys_string: str
) -> List[str]:

    if not keys_string:

        raise HTTPException(
            status_code=400,
            detail="keys is required"
        )

    keys_string = keys_string.strip()

    # ========================================================
    # JSON ARRAY
    # ========================================================

    try:

        parsed = json.loads(
            keys_string
        )

        if isinstance(
                parsed,
                list
        ):

            keys = []

            for item in parsed:

                if isinstance(
                        item,
                        str
                ):

                    item = item.strip()

                    if item:

                        keys.append(
                            item
                        )

            if not keys:

                raise HTTPException(
                    status_code=400,
                    detail="keys array is empty"
                )

            return normalize_requested_keys(
                keys
            )

    except json.JSONDecodeError:

        pass

    # ========================================================
    # COMMA SEPARATED
    # ========================================================

    keys = [
        item.strip()
        for item in keys_string.split(",")
        if item.strip()
    ]

    if not keys:

        raise HTTPException(
            status_code=400,
            detail="No valid keys found"
        )

    return normalize_requested_keys(
        keys
    )


# ============================================================
# OCR API
# ============================================================

@app.post("/extract")
async def extract(

        image: UploadFile = File(...),

        keys: str = Form(...)

):

    # ========================================================
    # VALIDATE IMAGE
    # ========================================================

    if not image.filename:

        raise HTTPException(
            status_code=400,
            detail="Image file is required"
        )

    # ========================================================
    # PARSE REQUESTED KEYS
    # ========================================================

    requested_keys = parse_keys(
        keys
    )

    print("==============================================")
    print("REQUESTED KEYS")
    print("==============================================")

    print(
        json.dumps(
            requested_keys,
            indent=4,
            ensure_ascii=False
        )
    )

    # ========================================================
    # ALLOWED EXTENSIONS
    # ========================================================

    allowed_extensions = {

        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"

    }

    extension = os.path.splitext(
        image.filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG, PNG, BMP "
                "and WEBP images are supported"
            )
        )

    # ========================================================
    # CREATE TEMPORARY FILE
    # ========================================================

    filename = (
        str(uuid.uuid4())
        + extension
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    print("==============================================")
    print("New OCR Request")
    print("Original File:", image.filename)
    print("Temporary File:", image_path)
    print("==============================================")

    try:

        # ====================================================
        # SAVE IMAGE WITH SIZE LIMIT
        # ====================================================

        total_size = 0

        with open(
                image_path,
                "wb"
        ) as buffer:

            while True:

                chunk = await image.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_size += len(
                    chunk
                )

                if total_size > MAX_FILE_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail="Image size cannot exceed 10 MB"
                    )

                buffer.write(
                    chunk
                )

        print(
            "Image saved successfully."
        )

        print(
            "Image size:",
            total_size,
            "bytes"
        )

        # ====================================================
        # OCR
        # ====================================================

        lines = extract_text_from_image(
            image_path
        )

        # ====================================================
        # DYNAMIC KEY-VALUE EXTRACTION
        # ====================================================

        key_values = extract_requested_key_values(
            lines,
            requested_keys
        )

        # ====================================================
        # RESPONSE
        # ====================================================

        response = {

            "success": True,

            "filename": image.filename,

            "requested_keys": requested_keys,

            "data": key_values,

            "raw_text": lines

        }

        # ====================================================
        # PRINT RESPONSE
        # ====================================================

        print("==============================================")
        print("FINAL OCR RESPONSE")
        print("==============================================")

        print(
            json.dumps(
                response,
                indent=4,
                ensure_ascii=False
            )
        )

        print("==============================================")

        return JSONResponse(
            status_code=200,
            content=response
        )

    except HTTPException:

        raise

    except Exception as e:

        print("==============================================")
        print("OCR ERROR")
        print("==============================================")

        print(
            str(e)
        )

        print("==============================================")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        # ====================================================
        # DELETE TEMP IMAGE
        # ====================================================

        if os.path.exists(
                image_path
        ):

            try:

                os.remove(
                    image_path
                )

                print(
                    "Temporary image deleted."
                )

            except Exception as e:

                print(
                    "Unable to delete temporary image:",
                    str(e)
                )

        try:

            await image.close()

        except Exception:

            pass


# ============================================================
# RUN APPLICATION
#
# Hugging Face:
# PORT = 7860
#
# IMPORTANT:
# Use ONE worker because PaddleOCR is memory heavy.
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        workers=1,
        reload=False
    )

