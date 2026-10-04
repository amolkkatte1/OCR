from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
from zoneinfo import ZoneInfo

import pytesseract
import io
import re
import threading
import uvicorn


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Test Parameter Generator API",
    description="Generate complete parameter JSON from test report images",
    version="3.0.0"
)


# ============================================================
# CONFIGURATION
# ============================================================

TIMEZONE = ZoneInfo("Asia/Kolkata")

TESSERACT_LANG = "eng"

TESSERACT_CONFIG = "--oem 3 --psm 6"

MAX_IMAGE_DIMENSION = 2500


# ============================================================
# PARAMETER ID GENERATION
# ============================================================

parameter_id_lock = threading.Lock()

last_parameter_id = 0


def generate_parameter_id():

    global last_parameter_id

    with parameter_id_lock:

        now = datetime.now(TIMEZONE)

        parameter_id = int(
            now.strftime("%Y%m%d%H%M%S")
            + f"{now.microsecond // 1000:03d}"
        )

        if parameter_id <= last_parameter_id:

            parameter_id = last_parameter_id + 1

        last_parameter_id = parameter_id

        return parameter_id


# ============================================================
# CURRENT DATETIME
# ============================================================

def get_current_datetime():

    return datetime.now(
        TIMEZONE
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image):

    image = image.convert("RGB")

    width, height = image.size

    # --------------------------------------------------------
    # Resize if required
    # --------------------------------------------------------

    if (
        width > MAX_IMAGE_DIMENSION
        or
        height > MAX_IMAGE_DIMENSION
    ):

        scale = min(
            MAX_IMAGE_DIMENSION / width,
            MAX_IMAGE_DIMENSION / height
        )

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

    # --------------------------------------------------------
    # Improve contrast
    # --------------------------------------------------------

    image = ImageEnhance.Contrast(
        image
    ).enhance(1.25)

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    image = image.filter(
        ImageFilter.SHARPEN
    )

    return image


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_ocr_text(text):

    if text is None:

        return ""

    text = text.replace(
        "\t",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# VALID OCR TEXT
# ============================================================

def is_valid_ocr_text(text):

    if not text:

        return False

    return bool(
        re.search(
            r"[A-Za-z0-9]",
            text
        )
    )


# ============================================================
# OCR DATA
# ============================================================

def extract_ocr_data(image: Image.Image):

    processed_image = preprocess_image(
        image
    )

    try:

        data = pytesseract.image_to_data(
            processed_image,
            lang=TESSERACT_LANG,
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT
        )

    except Exception as e:

        raise RuntimeError(
            f"Tesseract OCR failed: {str(e)}"
        )

    words = []

    total_items = len(
        data["text"]
    )

    for i in range(total_items):

        text = clean_ocr_text(
            data["text"][i]
        )

        if not text:

            continue

        if not is_valid_ocr_text(
            text
        ):

            continue

        try:

            confidence = float(
                data["conf"][i]
            )

        except Exception:

            confidence = -1

        # Ignore extremely low confidence text
        if confidence >= 0 and confidence < 10:

            continue

        words.append({

            "text":
                text,

            "x":
                int(data["left"][i]),

            "y":
                int(data["top"][i]),

            "width":
                int(data["width"][i]),

            "height":
                int(data["height"][i]),

            "conf":
                confidence
        })

    return words


# ============================================================
# GROUP OCR WORDS INTO ROWS
# ============================================================

def group_words_into_rows(words):

    if not words:

        return []

    words = sorted(
        words,
        key=lambda item: (
            item["y"],
            item["x"]
        )
    )

    rows = []

    for word in words:

        word_center_y = (
            word["y"]
            +
            word["height"] / 2
        )

        matched_row = None

        for row in rows:

            tolerance = max(
                8,
                row["average_height"] * 0.65
            )

            if abs(
                word_center_y
                -
                row["center_y"]
            ) <= tolerance:

                matched_row = row

                break

        if matched_row is not None:

            matched_row["words"].append(
                word
            )

            count = len(
                matched_row["words"]
            )

            matched_row["center_y"] = (

                (
                    matched_row["center_y"]
                    *
                    (count - 1)
                )
                +
                word_center_y

            ) / count

            matched_row["average_height"] = (

                sum(
                    item["height"]
                    for item
                    in matched_row["words"]
                )
                /
                count
            )

        else:

            rows.append({

                "center_y":
                    word_center_y,

                "average_height":
                    word["height"],

                "words":
                    [word]
            })

    rows.sort(
        key=lambda row: row["center_y"]
    )

    for row in rows:

        row["words"].sort(
            key=lambda item: item["x"]
        )

    return rows


# ============================================================
# COLUMN TEXT
# ============================================================

def get_column_text(words):

    if not words:

        return ""

    words = sorted(
        words,
        key=lambda item: item["x"]
    )

    return clean_ocr_text(
        " ".join(
            word["text"]
            for word in words
        )
    )


# ============================================================
# DETECT FOUR COLUMNS
# ============================================================

def detect_columns(
    row,
    image_width
):
    """
    Four columns:

        1 = Parameter Name
        2 = Value
        3 = Unit
        4 = Reference Range
    """

    column_1 = []
    column_2 = []
    column_3 = []
    column_4 = []

    # --------------------------------------------------------
    # Column boundaries
    # --------------------------------------------------------

    boundary_1 = image_width * 0.45

    boundary_2 = image_width * 0.65

    boundary_3 = image_width * 0.78

    for word in row["words"]:

        center_x = (
            word["x"]
            +
            word["width"] / 2
        )

        if center_x < boundary_1:

            column_1.append(
                word
            )

        elif center_x < boundary_2:

            column_2.append(
                word
            )

        elif center_x < boundary_3:

            column_3.append(
                word
            )

        else:

            column_4.append(
                word
            )

    return (
        column_1,
        column_2,
        column_3,
        column_4
    )


# ============================================================
# RANGE PARSER
# ============================================================

def parse_range(text):

    if not text:

        return {

            "lowerRange":
                None,

            "parameterRange":
                None,

            "upperRange":
                None
        }

    normalized = text

    normalized = normalized.replace(
        "–",
        "-"
    )

    normalized = normalized.replace(
        "—",
        "-"
    )

    normalized = normalized.replace(
        "−",
        "-"
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized
    ).strip()

    # --------------------------------------------------------
    # Example:
    #
    # 12 - 17
    # 12.0 - 17.0
    # --------------------------------------------------------

    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)",
        normalized
    )

    if match:

        lower = float(
            match.group(1)
        )

        upper = float(
            match.group(2)
        )

        return {

            "lowerRange":
                lower,

            "parameterRange":
                normalized,

            "upperRange":
                upper
        }

    # --------------------------------------------------------
    # Example:
    #
    # 12 to 17
    # --------------------------------------------------------

    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)",
        normalized,
        re.IGNORECASE
    )

    if match:

        lower = float(
            match.group(1)
        )

        upper = float(
            match.group(2)
        )

        return {

            "lowerRange":
                lower,

            "parameterRange":
                normalized,

            "upperRange":
                upper
        }

    # --------------------------------------------------------
    # > 5
    # --------------------------------------------------------

    match = re.search(
        r">\s*(-?\d+(?:\.\d+)?)",
        normalized
    )

    if match:

        value = float(
            match.group(1)
        )

        return {

            "lowerRange":
                value,

            "parameterRange":
                normalized,

            "upperRange":
                None
        }

    # --------------------------------------------------------
    # < 10
    # --------------------------------------------------------

    match = re.search(
        r"<\s*(-?\d+(?:\.\d+)?)",
        normalized
    )

    if match:

        value = float(
            match.group(1)
        )

        return {

            "lowerRange":
                None,

            "parameterRange":
                normalized,

            "upperRange":
                value
        }

    return {

        "lowerRange":
            None,

        "parameterRange":
            normalized,

        "upperRange":
            None
    }


# ============================================================
# PARSE ONE ROW
# ============================================================

def parse_ocr_row(
    row,
    image_width
):
    """
    Convert one image row into:

        parameterName
        value
        unit
        reference range
    """

    (
        column_1,
        column_2,
        column_3,
        column_4
    ) = detect_columns(
        row,
        image_width
    )

    parameter_name = get_column_text(
        column_1
    )

    value = get_column_text(
        column_2
    )

    unit = get_column_text(
        column_3
    )

    range_text = get_column_text(
        column_4
    )

    # --------------------------------------------------------
    # If column 4 was not detected correctly,
    # find range from row text.
    # --------------------------------------------------------

    if not range_text:

        full_row_text = get_column_text(
            row["words"]
        )

        match = re.search(
            r"(-?\d+(?:\.\d+)?)\s*[-–—−]\s*(-?\d+(?:\.\d+)?)",
            full_row_text
        )

        if match:

            range_text = match.group(0)

    range_data = parse_range(
        range_text
    )

    return {

        "parameterName":
            parameter_name,

        "value":
            value,

        "unit":
            unit,

        "lowerRange":
            range_data["lowerRange"],

        "parameterRange":
            range_data["parameterRange"],

        "upperRange":
            range_data["upperRange"]
    }


# ============================================================
# CHECK HEADER ROW
# ============================================================

def is_header_row(row):

    full_text = get_column_text(
        row["words"]
    ).lower()

    full_text = re.sub(
        r"\s+",
        " ",
        full_text
    ).strip()

    # --------------------------------------------------------
    # Common headers
    # --------------------------------------------------------

    headers = [

        "test name result unit reference range",

        "test name result unit reference",

        "parameter value unit reference range",

        "parameter result unit reference range",

        "test result unit reference range",

        "investigation result unit reference range",

        "test name",

        "parameter name",

        "parameter result",

        "reference range"
    ]

    for header in headers:

        if full_text == header:

            return True

    # --------------------------------------------------------
    # Detect header by keywords
    # --------------------------------------------------------

    has_test = (
        "test" in full_text
        or
        "parameter" in full_text
        or
        "investigation" in full_text
    )

    has_result = (
        "result" in full_text
        or
        "value" in full_text
    )

    has_unit = (
        "unit" in full_text
    )

    has_range = (
        "range" in full_text
        or
        "reference" in full_text
    )

    if (
        has_test
        and
        has_result
        and
        has_unit
        and
        has_range
    ):

        return True

    return False


# ============================================================
# VALID PARAMETER ROW
# ============================================================

def is_valid_parameter_row(
    parsed_row
):

    parameter_name = (
        parsed_row["parameterName"]
    )

    if not parameter_name:

        return False

    # --------------------------------------------------------
    # Ignore obvious headers
    # --------------------------------------------------------

    name = parameter_name.lower()

    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    ignored = {

        "parameter",

        "parameters",

        "test",

        "test name",

        "investigation",

        "investigations",

        "name",

        "result",

        "results",

        "value",

        "unit",

        "reference",

        "reference range",

        "normal range"
    }

    if name in ignored:

        return False

    return True


# ============================================================
# EXTRACT TABLE ROWS
# ============================================================

def extract_table_rows(
    image: Image.Image
):

    processed_image = preprocess_image(
        image
    )

    image_width, image_height = (
        processed_image.size
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    words = extract_ocr_data(
        processed_image
    )

    if not words:

        return []

    # --------------------------------------------------------
    # Group into rows
    # --------------------------------------------------------

    rows = group_words_into_rows(
        words
    )

    table_rows = []

    previous_parameter = None

    for row in rows:

        # ----------------------------------------------------
        # Skip table header
        # ----------------------------------------------------

        if is_header_row(row):

            continue

        # ----------------------------------------------------
        # Parse row
        # ----------------------------------------------------

        parsed_row = parse_ocr_row(
            row,
            image_width
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not is_valid_parameter_row(
            parsed_row
        ):

            continue

        parameter_name = (
            parsed_row["parameterName"]
        )

        # ----------------------------------------------------
        # Avoid duplicate consecutive rows
        # ----------------------------------------------------

        if (
            previous_parameter
            and
            parameter_name.lower()
            ==
            previous_parameter.lower()
        ):

            continue

        table_rows.append(
            parsed_row
        )

        previous_parameter = (
            parameter_name
        )

    return table_rows


# ============================================================
# CREATE COMPLETE PARAMETER OBJECT
# ============================================================

def create_parameter_object(
    parameter_name,
    value,
    unit,
    lower_range,
    parameter_range,
    upper_range,
    code,
    sequence
):

    current_datetime = (
        get_current_datetime()
    )

    return {

        # ----------------------------------------------------
        # BASIC
        # ----------------------------------------------------

        "parameterId":
            generate_parameter_id(),

        "parameterName":
            parameter_name,

        "code":
            code,

        "value":
            value,

        "sequence":
            sequence,

        "dataType":
            "String",

        "unit":
            unit,

        # ----------------------------------------------------
        # CRITERIA
        # ----------------------------------------------------

        "criteria":
            "Normal",

        "defaultVlue":
            "",

        "formula":
            "",

        # ----------------------------------------------------
        # RANGE
        # ----------------------------------------------------

        "upperRange":
            upper_range,

        "lowerRange":
            lower_range,

        "extrimUpperRange":
            0.0,

        "extrimLowerRange":
            0.0,

        "lowerAgeRange":
            0.0,

        "upperAgeRange":
            0.0,

        # ----------------------------------------------------
        # METHOD
        # ----------------------------------------------------

        "method":
            "",

        "context":
            "",

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        "isHideLable":
            False,

        "isHideLableOnRemport":
            False,

        "isLocalDictonery":
            False,

        "isWrapper":
            False,

        "isCalculative":
            False,

        "isImageResize":
            False,

        "isBold":
            True,

        "isNameBold":
            False,

        "isDescriptionParameter":
            False,

        # ----------------------------------------------------
        # OTHER
        # ----------------------------------------------------

        "position":
            None,

        "parameterRange":
            parameter_range,

        "isValueRequired":
            True,

        "lineCount":
            None,

        "isValueDiscription":
            False,

        # ----------------------------------------------------
        # AUDIT
        # ----------------------------------------------------

        "createdBy":
            101,

        "updatedBy":
            102,

        "createdAt":
            current_datetime,

        "updatedAt":
            current_datetime
    }


# ============================================================
# GENERATE FINAL PARAMETER LIST
# ============================================================

def generate_parameter_list(
    table_rows,
    code
):

    parameters = []

    sequence = 3

    for row in table_rows:

        parameter = create_parameter_object(

            parameter_name=
                row["parameterName"],

            value=
                row["value"],

            unit=
                row["unit"],

            lower_range=
                row["lowerRange"],

            parameter_range=
                row["parameterRange"],

            upper_range=
                row["upperRange"],

            code=
                code,

            sequence=
                sequence
        )

        parameters.append(
            parameter
        )

        sequence += 1

    return parameters


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status":
            "UP",

        "service":
            "Test Parameter Generator API",

        "mainEndpoint":
            "/api/parameters/generate"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "UP"
    }


# ============================================================
# MAIN API
# ============================================================

@app.post(
    "/api/parameters/generate"
)
async def generate_parameters_api(

    image: UploadFile = File(...),

    code: str = Form(...)
):

    # ========================================================
    # VALIDATE CODE
    # ========================================================

    if code is None:

        raise HTTPException(
            status_code=400,
            detail="code is required"
        )

    code = code.strip()

    if not code:

        raise HTTPException(
            status_code=400,
            detail="code cannot be empty"
        )

    # ========================================================
    # VALIDATE IMAGE
    # ========================================================

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="image is required"
        )

    if not image.filename:

        raise HTTPException(
            status_code=400,
            detail="Image filename is missing"
        )

    # ========================================================
    # READ IMAGE
    # ========================================================

    try:

        image_bytes = await image.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty"
            )

        pil_image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        pil_image.load()

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {str(e)}"
        )

    # ========================================================
    # OCR / TABLE EXTRACTION
    # ========================================================

    try:

        table_rows = extract_table_rows(
            pil_image
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # ========================================================
    # NO DATA
    # ========================================================

    if not table_rows:

        return JSONResponse(
            status_code=200,
            content=[]
        )

    # ========================================================
    # GENERATE COMPLETE JSON OBJECTS
    # ========================================================

    parameters = generate_parameter_list(
        table_rows=table_rows,
        code=code
    )

    # ========================================================
    # RETURN FULL JSON ARRAY
    # ========================================================

    return JSONResponse(
        status_code=200,
        content=parameters
    )


# ============================================================
# DEBUG API
# ============================================================

@app.post(
    "/api/parameters/generate/debug"
)
async def generate_parameters_debug_api(

    image: UploadFile = File(...),

    code: str = Form(...)
):

    if code is None or not code.strip():

        raise HTTPException(
            status_code=400,
            detail="code is required"
        )

    code = code.strip()

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        image_bytes = await image.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty"
            )

        pil_image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        pil_image.load()

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {str(e)}"
        )

    # --------------------------------------------------------
    # Extract table
    # --------------------------------------------------------

    try:

        table_rows = extract_table_rows(
            pil_image
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # --------------------------------------------------------
    # Generate parameters
    # --------------------------------------------------------

    parameters = generate_parameter_list(
        table_rows=table_rows,
        code=code
    )

    # --------------------------------------------------------
    # Debug response
    # --------------------------------------------------------

    return {

        "code":
            code,

        "rows":
            table_rows,

        "parameterCount":
            len(parameters),

        "parameters":
            parameters
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )