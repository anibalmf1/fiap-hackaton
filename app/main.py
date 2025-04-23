import os
import time
import shutil
from pathlib import Path
import threading
import requests

from fastapi import FastAPI, UploadFile, Form, Query
import aiofiles
from pydantic import BaseModel

from app.model import ModelHandler
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = FastAPI()
handler = ModelHandler()

os.makedirs("data", exist_ok=True)

# Directory to watch for new files
WATCH_DIR = r"C:\Users\aniba\Videos\hackaton\listener"
# Directory for processed files
PROCESSED_DIR = os.path.join(WATCH_DIR, "processed")

# Create directories if they don't exist
os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            # Wait a moment to ensure the file is fully written
            time.sleep(1)
            src_path = event.src_path
            filename = os.path.basename(src_path)

            # Webhook URL for n8n
            webhook_url = "http://localhost:5678/webhook/1e2a00f8-55df-4b89-82e6-56cac72b40a0"

            try:
                # Send POST request to the webhook with filename as parameter
                response = requests.post(webhook_url, json={"filename": filename})
                if response.status_code == 200:
                    print(f"File {filename} notification sent to n8n webhook successfully")
                else:
                    print(f"Error sending notification to webhook: Status code {response.status_code}")
            except Exception as e:
                print(f"Error sending notification to webhook: {e}")

def start_file_watcher():
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()
    print(f"Started watching directory: {WATCH_DIR}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

class FilenameRequest(BaseModel):
    filename: str


# Start the file watcher in a background thread
@app.on_event("startup")
def on_startup():
    thread = threading.Thread(target=start_file_watcher, daemon=True)
    thread.start()
    print("File watcher started in background")


@app.post("/train")
async def train(video: UploadFile, label: str = Form(...)):
    path = f"data/{video.filename}"
    async with aiofiles.open(path, "wb") as f:
        await f.write(await video.read())
    handler.train([path], label)
    return {"status": "success"}

@app.post("/predict")
async def predict(video: UploadFile):
    path = f"data/{video.filename}"
    async with aiofiles.open(path, "wb") as f:
        await f.write(await video.read())
    result = handler.predict(path)
    return {"result": result}

@app.post("/predict_filename")
async def predict_filename(request: FilenameRequest):
    try:
        file_path = os.path.join(WATCH_DIR, request.filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File {request.filename} not found in {WATCH_DIR}"}

        result = handler.predict(file_path)
        return {"result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/processed")
async def processed(request: FilenameRequest):
    try:
        src_path = os.path.join(WATCH_DIR, request.filename)
        dst_path = os.path.join(PROCESSED_DIR, request.filename)

        # Check if the source file exists
        if not os.path.exists(src_path):
            return {"status": "error", "message": f"File {request.filename} not found in {WATCH_DIR}"}

        # Move the file to the processed directory
        shutil.move(src_path, dst_path)

        return {"status": "success", "message": f"File {request.filename} moved to {PROCESSED_DIR}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
