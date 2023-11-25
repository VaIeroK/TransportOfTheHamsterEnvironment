import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import math
import csv
import yaml


def convert_coordinates(points, video_size):
    return [(int(x * video_size[0]), int(y * video_size[1])) for x, y in points]


def calculate_speed(current_detection, prev_detection, fps):
    d_pixels = math.sqrt((current_detection[0] - prev_detection[0]) ** 2 +
                         (current_detection[1] - prev_detection[1]) ** 2)

    object_width_pixels = current_detection[2] - current_detection[0]
    object_width_meters = 2.0
    object_width_frame = object_width_meters / object_width_pixels
    d_meters = d_pixels * object_width_frame
    speed_mps = d_meters * fps
    speed_kmph = speed_mps * 3.6

    return speed_kmph


def apply_mask(frame, points):
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame, mask


def save_to_csv(results_list, csv_file, video_file, current_time, area_index):
    with open('yolov8n_openvino_model/metadata.yaml', 'r') as file:
        metadata = yaml.load(file, Loader=yaml.FullLoader)

    names_dict = {idx: name for idx, name in metadata['names'].items()}

    transport_info = {vehicle_type: {'count': 0, 'total_speed': 0} for vehicle_type in metadata['names'].values()}

    previous_locations = {}

    for i, detection in enumerate(results_list[0].boxes.xyxy):
        try:
            x1, y1, x2, y2 = map(int, detection[:4])
            vehicle_class_id = results_list[0].boxes.cls[i].item()
            vehicle_type = names_dict.get(vehicle_class_id, 'Unknown')

            if vehicle_type == 'truck':
                vehicle_type = 'van'

            if vehicle_type in metadata['names'].values():
                if vehicle_type not in transport_info:
                    transport_info[vehicle_type] = {'count': 0, 'total_speed': 0}

                transport_info[vehicle_type]['count'] += 1

                if i in previous_locations:
                    speed = calculate_speed([x1, y1, x2, y2], previous_locations[i], cap.get(cv2.CAP_PROP_FPS))
                    transport_info[vehicle_type]['total_speed'] += speed

                previous_locations[i] = [x1, y1, x2, y2]

        except (ValueError, TypeError) as e:
            print(f"Error processing detection: {e}")
            continue

    data = [
        video_file,
        transport_info.get('car', {'count': 0, 'total_speed': 0})['count'],
        transport_info.get('car', {'count': 0, 'total_speed': 0})['total_speed'] /
        transport_info.get('car', {'count': 0, 'total_speed': 0})['count'] if
        transport_info.get('car', {'count': 0, 'total_speed': 0})['count'] > 0 else 0,
        transport_info.get('truck', {'count': 0, 'total_speed': 0})['count'],
        transport_info.get('truck', {'count': 0, 'total_speed': 0})['total_speed'] /
        transport_info.get('truck', {'count': 0, 'total_speed': 0})['count'] if
        transport_info.get('truck', {'count': 0, 'total_speed': 0})['count'] > 0 else 0,
        transport_info.get('bus', {'count': 0, 'total_speed': 0})['count'],
        transport_info.get('bus', {'count': 0, 'total_speed': 0})['total_speed'] /
        transport_info.get('bus', {'count': 0, 'total_speed': 0})['count'] if
        transport_info.get('bus', {'count': 0, 'total_speed': 0})['count'] > 0 else 0
    ]

    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(data)

    print(f"Video {video_file}, area {area_index + 1} processed.")


model = YOLO('yolov8n.pt')
model.export(format='openvino')

ov_model = YOLO('yolov8n_openvino_model/')

video_folder = "D:/PythonProjects/video/Video1"
json_folder = "D:/PythonProjects/jsons"
results_folder = "D:/PythonProjects/results/"

os.makedirs(results_folder, exist_ok=True)

video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

for video_file in video_files:
    name = video_file.rsplit('.', 1)[0]
    json_file = name + ".json"
    if json_file in json_files:
        video_path = os.path.join(video_folder, video_file)
        json_path = os.path.join(json_folder, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        cap = cv2.VideoCapture(video_path)
        video_size = (int(cap.get(3)), int(cap.get(4)))

        with open('yolov8n_openvino_model/metadata.yaml', 'r') as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)

        for i, area in enumerate(data["areas"]):
            points = convert_coordinates(area, video_size)

            cap = cv2.VideoCapture(video_path)

            previous_locations = {}
            current_time = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                masked_frame, _ = apply_mask(frame, points)
                results_list = ov_model(masked_frame)

                for i, detection in enumerate(results_list[0].boxes.xyxy):
                    x1, y1, x2, y2 = map(int, detection[:4])

                    cv2.putText(masked_frame, f"ID: {i + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                1)

                    if i in previous_locations:
                        speed = calculate_speed([x1, y1, x2, y2], previous_locations[i], cap.get(cv2.CAP_PROP_FPS))
                        cv2.putText(masked_frame, f"Speed: {speed:.2f} km/h", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 1)

                    cv2.rectangle(masked_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    previous_locations[i] = [x1, y1, x2, y2]

                cv2.imshow("YOLOv8 OpenVINO Inference", masked_frame)

                key = cv2.waitKey(1)
                if key & 0xFF == ord("q") or key & 0xFF == ord("й"):
                    break

                save_to_csv(results_list, os.path.join(results_folder, "results.csv"), video_file, current_time, i)

            cap.release()
            cv2.destroyAllWindows()

            print(f"Video {video_file} processed.")
    else:
        print(f"Data file {json_file} not found for video {video_file}.")
