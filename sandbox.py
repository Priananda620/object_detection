import config
from Token import FlusonicToken
import queue
import threading
from Yolo import Yolov4
import cv2
import time
from collections import defaultdict
import numpy as np
import math
from collections import Counter

from ultralytics import YOLO
from ultralytics import NAS
from ultralytics.solutions import object_counter
from ultralytics.solutions import speed_estimation

FPS=20
MAX_QUEUE_SIZE = FPS*3
FRAME_DISPLAY_INTERVAL = 1 / FPS
QUEUE_SIZE_MINIMUM_THRESHOLD = (FPS)

def render_text(text, left_box, frame, font_scale = .5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    cv2.rectangle(
        frame,
        left_box,
        (left_box[0] + text_w, left_box[1] + text_h + 2),
        (255, 255, 255),
        thickness=cv2.FILLED
    )
    
    cv2.putText(
        frame,
        text,
        (left_box[0], int(left_box[1] + text_h)),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness,
    )


def get_polyline_dist(polyline_points, frame, box_centroid_x, box_centroid_y, box_w, box_h, box_class):

    box_half_w = box_w // 2
    box_half_h = box_h // 2

    box_top_left_x=int(box_centroid_x-box_half_w)
    box_top_left_y=int(box_centroid_y-box_half_h)
    top_left_box = (box_top_left_x, box_top_left_y)

    box_half_diagonal_distance = (math.sqrt((box_centroid_x - top_left_box[0])**2 + (box_centroid_y - top_left_box[1])**2)-4)
    box_quart_diagonal_distance = box_half_diagonal_distance // 2
    box_quart_w = box_half_w // 2
    box_quart_h = box_half_h // 2

    box_top_right_x=int(box_centroid_x+box_half_w)
    box_top_right_y=int(box_centroid_y-box_half_h)
    top_right_box = (box_top_right_x, box_top_right_y)

    box_bottom_right_x=int(box_centroid_x+box_half_w)
    box_bottom_right_y=int(box_centroid_y+box_half_h)
    bottom_right_box = (box_bottom_right_x, box_bottom_right_y)

    box_bottom_left_x=int(box_centroid_x-box_half_w)
    box_bottom_left_y=int(box_centroid_y+box_half_h)
    bottom_left_box = (box_bottom_left_x, box_bottom_left_y)

    cv2.circle(frame, (top_left_box), radius=2, color=(255, 0, 0), thickness=2)
    cv2.circle(frame, (top_right_box), radius=2, color=(255, 0, 0), thickness=2)
    cv2.circle(frame, (bottom_left_box), radius=2, color=(255, 0, 0), thickness=2)
    cv2.circle(frame, (bottom_right_box), radius=2, color=(255, 0, 0), thickness=2)

    first_point = polyline_points[0, 0]
    last_point = polyline_points[-1, 0]

    # delta_x = last_point[0] - first_point[0]
    # delta_y = last_point[1] - first_point[1]
    delta_x = abs(last_point[0] - first_point[0]) <=12
    delta_y = abs(last_point[1] - first_point[1]) <= 12
    distance = np.floor(np.linalg.norm(first_point - last_point))


    object_is_stationary = distance <= 5 and distance < box_quart_h

    if delta_x:
        delta = "up."
        dist_is_outside_box = distance>=box_half_h
        dist_is_half_box = ((distance>box_quart_h) and not dist_is_outside_box)
        dist_is_quart_box = ((distance<box_quart_h) and not dist_is_outside_box)
    elif delta_y:
        delta = "hor."
        dist_is_outside_box = distance>=box_half_w
        dist_is_half_box = ((distance>box_quart_w) and not dist_is_outside_box)
        dist_is_quart_box = ((distance<box_quart_w) and not dist_is_outside_box)
    else:
        delta = "diag."
        dist_is_outside_box = distance>=box_half_diagonal_distance
        dist_is_half_box = ((distance>box_quart_diagonal_distance) and not dist_is_outside_box)
        dist_is_quart_box = ((distance<box_quart_diagonal_distance) and not dist_is_outside_box)
    
    render_text(f'dist:{distance}|{delta}|{box_class}', top_left_box, frame)
    render_text(f'({'T' if dist_is_outside_box else 'F'}|{'T' if dist_is_half_box else 'F'}|{'T' if dist_is_quart_box else 'F'})|is stop:{object_is_stationary}', bottom_left_box, frame)

    return distance, delta, dist_is_outside_box, dist_is_half_box, dist_is_quart_box, object_is_stationary


def read_frames(model_obj, CLASS_NAMES, only_classes):
    global frame_queue

    total_frames = 0
    mod20 = 0
    detected_classes = []
    max_classes_count_per_20frames = {}
    
    track_history = defaultdict(lambda: [])
    while True:
        ret, frame = stream.read()
        if not ret:
            break

        total_frames += 1

        height, width, _ = frame.shape

        screen_width = 1280
        screen_height = 720
        scaling_factor = min(screen_width / width, screen_height / height)
        resized_width = int(width * scaling_factor)
        resized_height = int(height * scaling_factor)
        resized_frame = cv2.resize(frame, (resized_width, resized_height))

        if len(only_classes) > 0:
            tracks = model_obj.track(resized_frame, persist=True, classes=only_classes)
        else:
            tracks = model_obj.track(resized_frame, persist=True)

        if total_frames%20 == 0 and not(total_frames == 0):
            total_frames=0
            mod20+=1

            for sublist in detected_classes:
                class_counts = {str(class_num): sublist.count(class_num) for class_num in set(sublist)}
                for class_num, count in class_counts.items():
                    if class_num in max_classes_count_per_20frames:
                        max_classes_count_per_20frames[class_num] = max(max_classes_count_per_20frames[class_num], count)
                    else:
                        max_classes_count_per_20frames[class_num] = count
            # print(f'{max_classes_count_per_20frames}')
            detected_classes = []
        render_text(f"MOD 20 = {mod20}", (0,0),resized_frame, font_scale=1.5)
        render_text(f'{max_classes_count_per_20frames}', (0,50),resized_frame, font_scale=1.5)

        if tracks[0].boxes is not None:
            boxes = tracks[0].boxes.xywh.cpu()
            track_ids = tracks[0].boxes.id.int().cpu().tolist() if tracks[0].boxes.id is not None else []
            box_classes = tracks[0].boxes.cls.int().cpu().tolist() 
            annotated_frame = tracks[0].plot()

            detected_class_inframe = []

            for box, track_id, box_class in zip(boxes, track_ids, box_classes):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 20:
                    track.pop(0)

                cv2.circle(annotated_frame, (int(x), int(y)), radius=2, color=(255, 255, 0), thickness=2)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 255), thickness=2)

                detected_class_inframe.append(box_class)

                distance, delta, dist_is_outside_box, dist_is_half_box, dist_is_quart_box, object_is_stationary = get_polyline_dist(points, annotated_frame, x, y, w, h, CLASS_NAMES[box_class])

            class_counts = {str(class_num): detected_class_inframe.count(class_num) for class_num in set(detected_class_inframe)}
            counts = ', '.join(f"{class_num}:{count}" for class_num, count in class_counts.items())
            render_text(f"{counts}", (0,100),annotated_frame, font_scale=1.5)

            detected_classes.append(detected_class_inframe)
            # print(detected_classes)

        frame_queue.put(annotated_frame)
        
if __name__ == "__main__":
    # model_obj = Yolov4()

    # model = YOLO('rtdetr-l.pt')
    # model = YOLO('yolov8_best_visdrone.pt')
    model = YOLO('yolov8n-visdrone.pt')

    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image
    # only_classes = [2, 5, 7]# YOLO
    only_classes = []
    CLASS_NAMES = model.names 
    
    token_generator = FlusonicToken('Kembangan-Selatan-003', 10).get_tokenized_url()
    stream_url = token_generator
    print(stream_url)
    print(CLASS_NAMES)

    # time.sleep(3)

    # GBK-003
    # JPO-Merdeka-Barat-006
    # Sunter-Jaya-002  Bali-Mester-006 Gunung-004 Cipinang-Cempedak-001/2 Duri-Kosambi-002 Gambir-006 Gambir-011 Gelora-001 Gelora-009 Kayu-Putih-003 Kembangan-Selatan-001 Kembangan-Selatan-006/3 Kenari-006 Paseban-001 Pulo-Gadung-001 Kuningan-Timur-002 Kramat-004 Paseban-004
    # horizontalBendungan-Hilir-005 Gambir-004 Bendungan-Hilir-005 Bendungan-Hilir-006
    # diagonalCeger-002  Pasar-Baru-009  Gunung-Sahari-Utara-007 Kenari-007/8 Pondok-Pinang-009
    # bestqualityPetojo-Utara-002 ***MP-Bangka-003***
    # Pospol-Merdeka-Utara car 
    # with parked carPasar-Baru-020 MP-Ancol-001
    #red light with moving cat Gelora-005 
    # from top Menteng-001 Menteng-005
    # redlight Paseban-004

    frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    stream = cv2.VideoCapture(stream_url)

    frame_reader_thread = threading.Thread(target=read_frames, args=(model, CLASS_NAMES, only_classes))
    frame_reader_thread.daemon = True
    frame_reader_thread.start()

    try:
        while True:
            start_time = time.time()
            current_size = frame_queue.qsize()

            if(current_size>=QUEUE_SIZE_MINIMUM_THRESHOLD):
                frame = frame_queue.get()
                cv2.imshow("cv2", frame)

            # print("Frame Queue Length:", current_size)

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


