import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
import shutil
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from werkzeug.utils import secure_filename
from starlette.responses import RedirectResponse, FileResponse

UPLOAD_FOLDER = "storage/uploads/"
DOWNLOAD_FOLDER = "storage/downloads/"
SRC_FOLDER = "src/"
DST_FOLDER = "dst/"
ALLOWED_EXTENSIONS = {".mp4"}

app = FastAPI()

app.mount("/storage/uploads", StaticFiles(directory="storage/uploads"), name="storage/uploads")
app.mount("/storage/downloads", StaticFiles(directory="storage/downloads"), name="storage/downloads")
app.mount("/storage/test_data", StaticFiles(directory="storage/test_data"), name="storage/test_data")


@app.get("/")
async def docs_redirect():
    """Launches start page. Just redirects to API documentation."""
    return RedirectResponse(url="/docs")


@app.post("/upload_src")
async def upload_src_video(uploaded_file: UploadFile = File(...)):
    """Uploads source video uploaded_file to the server UPLOAD_FOLDER/SRC_FOLDER using form data."""
    try:
        destination_path = f"{UPLOAD_FOLDER}{SRC_FOLDER}"
        return await upload_to_folder(uploaded_file, destination_path)
    except Exception as e:
        return {"message": f"There was an error uploading the file:{str(e)}"}
    finally:
        uploaded_file.file.close()


@app.post("/upload_dst")
async def upload_dst_video(uploaded_file: UploadFile = File(...)):
    """Uploads destination video uploaded_file to the server UPLOAD_FOLDER/DST_FOLDER using form data."""
    try:
        destination_path = f"{UPLOAD_FOLDER}{DST_FOLDER}"
        return await upload_to_folder(uploaded_file, destination_path)
    except Exception as e:
        return {"message": f"There was an error uploading the file:{str(e)}"}
    finally:
        uploaded_file.file.close()
    return {"message": f"Successfully uploaded {uploaded_file.filename}"}


@app.get("/process_videos/")
async def process_videos(src_video: str = '', dst_video: str = ''):
    return print(f"Source video: {src_video}, destination video: {dst_video}")


async def upload_to_folder(uploaded_file, destination_path):
    if not allowed_file(uploaded_file.filename):
        return {
            "message": "Not allowed file format: only .mp4 files are allowed."
        }
    filename = secure_filename(uploaded_file.filename)
    destination = Path(destination_path, filename)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return {"message": f"Successfully uploaded {uploaded_file.filename}"}


def allowed_file(filename):
    """Checks whether uploaded filename has an allowed extension."""
    return Path(filename).suffix in ALLOWED_EXTENSIONS


@app.get("/download/{filename}")
def download_final(filename):
    """Downloads final video with file name filename from DOWNLOAD_FOLDER if exists"""
    try:
        file_path = os.path.join(DOWNLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            return FileResponse(
                file_path, media_type="application/octet-stream", filename=filename
            )
    except Exception as e:
        return {"message": f"There was an error downloading the file:{str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
