import cv2

def extract_frames(video_path, num_frames=16):
    capture = cv2.VideoCapture(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total / num_frames) for i in range(num_frames)]
    frames = []
    for idx in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    return frames