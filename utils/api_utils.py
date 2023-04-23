import glob
import os
import shutil
from pathlib import Path
from werkzeug.utils import secure_filename
from utils.api_constants import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    ALLOWED_ARCHIVE_EXTENSIONS
)
from utils.enums import SourceType


async def upload_to_folder(uploaded_file, source_type, destination_path):
    if not allowed_file(uploaded_file.filename, source_type):
        return {
            "message": "Not allowed file format."
        }
    filename = secure_filename(uploaded_file.filename)
    destination = Path(destination_path, filename)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return {"message": f"Successfully uploaded {uploaded_file.filename}"}


def clean_folder(folder_path):
    """Cleans contents of the folder."""
    try:
        files = glob.glob(f"{folder_path}*")
        for f in files:
            os.remove(f)
    except Exception as e:
        return {
            "message":
                f"There was an error when cleaning the folder: {str(e)}"
        }


def allowed_file(filename, source_type):
    """Checks whether uploaded file has an allowed extension."""
    suffix = Path(filename).suffix.lower()
    allowed_extensions = ALLOWED_IMAGE_EXTENSIONS \
        if source_type == SourceType.IMAGE \
        else (
            ALLOWED_VIDEO_EXTENSIONS if source_type == SourceType.VIDEO
            else ALLOWED_ARCHIVE_EXTENSIONS)

    return suffix in allowed_extensions
