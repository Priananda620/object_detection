import cv2
import queue
import threading
import time
import numpy as np
from Yolo import Yolov4
from Token import FlusonicToken

from ultralytics import YOLO
from ultralytics.solutions import object_counter

# STREAM_URL = "https://cctv.molecool.id/Cempaka-Putih-Timur-013/video.m3u8?token=fc7ccef3f3fa41c2644c95bbde892519795a1ec9-caaadc601326886935f88e07a2c9b193-1709180184-1709176584"
FPS=20
MAX_QUEUE_SIZE = FPS*3
FRAME_DISPLAY_INTERVAL = 1 / FPS
QUEUE_SIZE_MINIMUM_THRESHOLD = (MAX_QUEUE_SIZE/2)


def read_frames(yolo_obj):
    global frame_queue
    while True:
        ret, frame = stream.read()
        if not ret:
            break

        height, width, _ = frame.shape
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(frame, (width // 2, height // 2))
        height, width, _ = resized_frame.shape

        outcome = yolo_obj.Inference(image=resized_frame,original_width=width,original_height=height)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded_frame = cv2.imencode('.jpg', outcome, encode_param)

        frame_queue.put(encoded_frame)
        
if __name__ == "__main__":
    yolo_obj = Yolov4()

    
    token_generator = FlusonicToken('Gambir-061', 10).get_tokenized_url()
    stream_url = token_generator

    frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    stream = cv2.VideoCapture(stream_url)

    frame_reader_thread = threading.Thread(target=read_frames, args=(yolo_obj,))
    frame_reader_thread.daemon = True
    frame_reader_thread.start()

    try:
        while True:
            start_time = time.time()
            current_size = frame_queue.qsize()

            if(current_size>=QUEUE_SIZE_MINIMUM_THRESHOLD):
                encoded_frame = frame_queue.get()
                frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
                cv2.imshow("cv2", frame)

            print("Frame Queue Length:", current_size)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
            processing_time = time.time() - start_time

            if processing_time < FRAME_DISPLAY_INTERVAL:
                time.sleep(FRAME_DISPLAY_INTERVAL - processing_time)
    except Exception as e:
        print("ERROR:", e)
    finally:
        # Release and close stream
        stream.release()
        cv2.destroyAllWindows()


