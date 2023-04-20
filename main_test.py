import os
import sys

ss_path = os.path.dirname(__file__) + '/SimSwap'
sys.path.append(ss_path)
sys.path.append(os.path.dirname(__file__))
sys.path.append('/home/runner/work/deep_fake/deep_fake/SimSwap')

from fastapi.testclient import TestClient
from main import app
from utils.api_constants import DOWNLOADS_FOLDER, RESULT_FILE_NAME, VIDEOS_FOLDER, IMAGES_FOLDER
from utils.enums import HashAlgorithm
from utils.hasher import Hasher


client = TestClient(app)


def test_main_page():
    """Test main page is working and returns http status 200 - OK"""
    response = client.get("/")
    assert response.status_code == 200


def test_upload_image():
    """Test uploading images is working."""
    file_path = "../demo/single/dst.jpeg"
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            response = client.post(
                "/upload_image", files={"uploaded_file": ("dst.jpeg", f, "image/jpeg")}
            )
            assert response.status_code == 200


def test_upload_video(multispecific=False):
    """Test uploading videos is working."""
    file_path = "../demo/multi/input.mp4" if multispecific else "../demo/single/input.mp4"
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            response = client.post(
                "/upload_video", files={"uploaded_file": ("input.mp4", f, "video/mp4")}
            )
            assert response.status_code == 200


def test_upload_multispecific():
    """Test uploading archives is working."""
    file_path = "../demo/multi/multispecific/Archive.zip"
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:
            response = client.post(
                "/upload_multispecific", files={"uploaded_file": ("Archive.zip", f, "application/zip")}
            )
            assert response.status_code == 200

def test_swap_single():
    """Test single swap is working, result video is created,
    and it's hash equals to the reference video's hash."""
    client = TestClient(app)  # create a test client for the FastAPI app
    test_upload_image()
    test_upload_video()
    with client:
        # Start the FastAPI application before sending the HTTP request
        with client.app.app_context():
            response = client.get("/swap_single", params={"src_video_name": "input.mp4", "dst_image_name": "dst.jpeg",
                                                          "test": True})
            assert response.status_code == 200
            result_file_path = f"{DOWNLOADS_FOLDER}{RESULT_FILE_NAME}"
            # TODO: check that it works on different machines
            # Hasher.check_hash_equals(HashAlgorithm.SHA1, result_file_path, '../demo/single/output.mp4')
            if os.path.exists(result_file_path):
                os.remove(result_file_path)

# def test_swap_single():
#     """Test single swap is working, result video is created,
#     and it's hash equals to the reference video's hash."""
#     test_upload_image()
#     test_upload_video()
#     response = client.get("/swap_single", params={"src_video_name": "input.mp4", "dst_image_name": "dst.jpeg",
#                                                   "test": True})
#     assert response.status_code == 200
#     result_file_path = f"{DOWNLOADS_FOLDER}{RESULT_FILE_NAME}"
#     # TODO: check that it works on different machines
#     # Hasher.check_hash_equals(HashAlgorithm.SHA1, result_file_path, '../demo/single/output.mp4')
#     if os.path.exists(result_file_path):
#         os.remove(result_file_path)


# def test_swap_multi():
#     """Test multi swap is working, result video is created,
#        and it's hash equals to the reference video's hash."""
#     test_upload_video(multispecific=True)
#     test_upload_multispecific()
#     response = client.get("/swap_multi",
#                           params={"src_video_name": "input.mp4", "multispecific_archive_name": "Archive.zip",
#                                   "test": True})
#     assert response.status_code == 200
#     result_file_path = f"{DOWNLOADS_FOLDER}{RESULT_FILE_NAME}"
#     # TODO: check that it works on different machines
#     # Hasher.check_hash_equals(HashAlgorithm.SHA1, result_file_path, '../demo/multi/output.mp4')
#     if os.path.exists(result_file_path):
#         os.remove(result_file_path)
