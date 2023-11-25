import os
import shutil

from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from starlette.responses import JSONResponse

upload_router = APIRouter()


def is_mp4(filename):
    return filename.lower().endswith('.mp4')


@upload_router.post("/upload_video/")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        if not is_mp4(file.filename):
            return JSONResponse(status_code=400, content={"message": "Only MP4 files are allowed"})

        #background_tasks.add_task(process_video, file_path)  # Добавляем задачу обработки в фоновые задачи
        return JSONResponse(status_code=201, content={"message": "File uploaded successfully. Processing started."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Failed to upload file: {e}"})
