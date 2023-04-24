import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse, FileResponse
from FaceDetectAntelopeModel import FaceDetectAntelopeModel
from SwapManager import SwapManager
from utils.api_constants import (
    DOWNLOADS_FOLDER,
    RESULT_FILE_NAME,
    VIDEOS_FOLDER,
    IMAGES_FOLDER,
    MULTISPECIFIC_FOLDER,
)
from utils.api_utils import upload_to_folder, clean_folder
from utils.enums import SourceType
from zipfile import ZipFile

app = FastAPI()

os.chdir("SimSwap")


@app.get("/")
async def docs_redirect():
    """Launches start page. Just redirects to API documentation."""
    return RedirectResponse(url="/docs")


@app.post("/upload_video")
async def upload_video(uploaded_file: UploadFile = File(...)):
    """Uploads destination video to the server VIDEOS_FOLDER using
    form data.
    """
    try:
        clean_folder(VIDEOS_FOLDER)
        return await upload_to_folder(
            uploaded_file,
            SourceType.VIDEO,
            VIDEOS_FOLDER
        )
    except Exception as e:
        return {"message": f"There was an error uploading the file:{str(e)}"}
    finally:
        uploaded_file.file.close()


@app.post("/upload_image")
async def upload_image(uploaded_file: UploadFile = File(...)):
    """Uploads destination face image to the server IMAGES_FOLDER
    using form data.
    """
    try:
        clean_folder(IMAGES_FOLDER)
        return await upload_to_folder(
            uploaded_file,
            SourceType.IMAGE,
            IMAGES_FOLDER
        )
    except Exception as e:
        return {"message": f"There was an error uploading the file:{str(e)}"}
    finally:
        uploaded_file.file.close()


@app.post("/upload_multispecific")
async def upload_multispecific(uploaded_file: UploadFile = File(...)):
    """Uploads archived multispecific folder to the server MULTISPECIFIC_FOLDER
    using form data.
    """
    try:
        clean_folder(MULTISPECIFIC_FOLDER)
        return await upload_to_folder(
            uploaded_file, SourceType.ARCHIVE, MULTISPECIFIC_FOLDER
        )
    except Exception as e:
        return {"message": f"There was an error uploading the file:{str(e)}"}
    finally:
        uploaded_file.file.close()


@app.get("/swap_single")
async def swap_single(
    src_video_name: str = "", dst_image_name: str = "", test: bool = False
):
    """Swaps single face found in source video with a face from destination
    image, and saves result video in DOWNLOADS_FOLDER. \n
    <b>Allowed image extensions:</b> .jpg, .jpeg, .zip. \n
    <b>Allowed video extensions:</b> .mp4. \n
    <b><font color='red'>IMPORTANT: Set <i>'test'</i> parameter to
    <i>'true'</i> only when running tests!</font></b>
    """
    try:
        clean_folder(DOWNLOADS_FOLDER)
        fd_model = FaceDetectAntelopeModel(0.6, 640)
        swap_manager = SwapManager(fd_model, test)

        return swap_manager.swap_single(
            f"{VIDEOS_FOLDER}{src_video_name}",
            f"{DOWNLOADS_FOLDER}{RESULT_FILE_NAME}",
            f"{IMAGES_FOLDER}{dst_image_name}",
        )
    except Exception as e:
        return {
            "message":
                f"There was an error when creating deep fake video: {str(e)}"
        }


@app.get("/swap_multi")
async def swap_multi(
    src_video_name: str = "",
    multispecific_archive_name: str = "",
    test: bool = False
):
    """Swaps multiple faces found in source video with DST_XX faces
    corresponding to SRC__XX faces found in MULTISPECIFIC_FOLDER, and saves
    result video in DOWNLOADS_FOLDER. \n
    Archive should contain SRC_XX.jpg(.jpeg/.png) and DST_XX.jpg(.jpeg/.png)
    image files. \n
    <b>Allowed video extensions:</b> .mp4. \n
    <b>Allowed archive extensions:</b> .zip. \n
    <b><font color='red'>IMPORTANT: Set <i>'test'</i> parameter to
    <i>'true'</i> only when running tests!</font></b>
    """
    try:
        clean_folder(DOWNLOADS_FOLDER)
        fd_model = FaceDetectAntelopeModel(0.6, 640)
        swap_manager = SwapManager(fd_model, test)

        with ZipFile(
            f"{MULTISPECIFIC_FOLDER}{multispecific_archive_name}", "r"
        ) as archive:
            archive.extractall(path=MULTISPECIFIC_FOLDER)

        return swap_manager.swap_multi(
            f"{VIDEOS_FOLDER}{src_video_name}",
            f"{DOWNLOADS_FOLDER}{RESULT_FILE_NAME}",
            f"{MULTISPECIFIC_FOLDER}",
        )
    except Exception as e:
        return {
            "message":
                f"There was an error when creating deep fake video: {str(e)}"
        }


@app.get("/download_result")
def download_result():
    """Downloads final video with the name RESULT_FILE_NAME filename
    from DOWNLOAD_FOLDER if the file exists
    """
    try:
        file_path = f"{DOWNLOADS_FOLDER}{RESULT_FILE_NAME}"
        if os.path.isfile(file_path):
            return FileResponse(
                file_path,
                media_type="application/octet-stream",
                filename=RESULT_FILE_NAME,
            )
        else:
            return {"message": f"{RESULT_FILE_NAME} was not found on server."}

    except Exception as e:
        return {
            "message":
                f"There was an error downloading {RESULT_FILE_NAME}: {str(e)}"
        }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
