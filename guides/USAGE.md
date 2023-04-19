# How to swap a single face in the video using API:
    1. Upload the destination video using POST method /upload_video. 
        Allowed extensions: .mp4.
    2. Upload the destination face image using POST method /upload_image.
        Allowed extensions: .jpg, .jpeg, .png.
    3. Call GET method /swap_single with the following parameters:
        src_video_name - name of your video file (e.g. input.mp4)
        dst_image_name - name of your face image file (e.g. dst.jpeg)
        test = False
    4. Download your result video using GET method /download.

# How to swap multiple faces in the video using API:
    1. Upload the destination video using POST method /upload_video.
        Allowed extensions: .mp4.
    2. Upload the archive with face images using POST method /upload_multispecific.
        Allowed extensions: .zip.
        The archive should contain numbered src and dst images, e.g.:
            |
            |--SRC_01.jpg
            |--DST_01.jpg
            |--SRC_02.jpg
            |--DST_02.jpg
    3. Call GET method /swap_multi with the following parameters:
        src_video_name - name of your video file (e.g. input.mp4)
        multispecific_archive_name - name of your archive file (e.g. Archive.zip)
        test = False
    4. Download your result video using GET method /download.

