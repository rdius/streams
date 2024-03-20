import cv2
import numpy as np
import time
import os

import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
np.random.seed(543210)

def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

class Detector1:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320) 
        self.net.setInputScale(1.0/127.5)                
        self.net.setInputMean((127.5, 127.5, 127.5)) 
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
            self.classesList.insert(0, '__Background__')
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))


    def detect_objects(self, frame):
        classLabelIDs, confidences, bboxes = self.net.detect(frame, confThreshold=0.5)

        if classLabelIDs is not None:
            for i in range(len(classLabelIDs)):
                classLabelID = classLabelIDs[i]
                classConfidence = confidences[i]
                bbox = bboxes[i]
                classLabel = self.classesList[classLabelID]
                classColor = [int(c) for c in self.colorList[classLabelID]]

                displayText = "{} {:.2f}".format(classLabel, classConfidence)

                x, y, w, h = bbox

                cv2.rectangle(frame, (x, y), (x + w, y + h), color=classColor, thickness=1)
                cv2.putText(frame, displayText, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
                #######
                

                ######
        return frame


def process_frame(frame, model):
    # Frame processing logic
    detections = model.detect_objects(frame)
    return detections

def run_tracker_in_thread_parallelized(videoPath, model, output_queue, video_output_path, camera_label, reduced_width=320, reduced_height=240, skip_rate=10, mse_threshold=1000, max_queue_size=30):
    video = cv2.VideoCapture(videoPath)
    executor = ThreadPoolExecutor(max_workers=2)  # Adjust based on your hardware

    frame_count = 0
    ret, prev_frame_processed = video.read()
    if ret:
        prev_frame_processed = cv2.resize(prev_frame_processed, (reduced_width, reduced_height))

    local_queue = Queue(maxsize=max_queue_size)

    while True:
        ret, frame = video.read()

        if not ret:
            print(f"End of video stream from {camera_label}")
            break

        frame = cv2.resize(frame, (reduced_width, reduced_height))

        if frame_count % skip_rate != 0:
            frame_count += 1
            continue

        frame_diff = mse(frame, prev_frame_processed)

        if frame_diff > mse_threshold:
            if local_queue.full():
                local_queue.get()

            local_queue.put(frame)

        if not local_queue.empty():
            frame_to_process = local_queue.get()

            # Submit frame processing tasks to the executor
            future = executor.submit(process_frame, frame_to_process, model)
            res_plotted = future.result()  # Get the processed frame result

            output_queue.put(res_plotted)
            prev_frame_processed = frame_to_process.copy()

        frame_count += 1

    video.release()
    executor.shutdown()

def main():
    rtsp_url1 = 'rtsp://admin:Copernilabs2024@192.168.1.105:554/cam/realmonitor?channel=1&subtype=0'
    rtsp_url2 = 'rtsp://admin:Copernilabs2024@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0'

    # YOLO model paths
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    # Create instances of Detector1 for both cameras
    detector1 = Detector1(rtsp_url1, configPath, modelPath, classesPath)
    detector2 = Detector1(rtsp_url2, configPath, modelPath, classesPath)

    # Run the detectors in separate threads
    queue1 = Queue()  # Create a queue for output frames
    queue2 = Queue()  # Create another queue for the second camera
    thread1 = threading.Thread(target=run_tracker_in_thread_parallelized, args=(rtsp_url1, detector1, queue1, "output1.mp4", "Camera 2", 320, 240, 10, 1000, 30), daemon=True)
    thread2 = threading.Thread(target=run_tracker_in_thread_parallelized, args=(rtsp_url2, detector2, queue2, "output2.mp4", "Camera 1", 320, 240, 10, 1000, 30), daemon=True)
    thread1.start()
    thread2.start()

    # Display frames from the queues
    while True:
        if not queue1.empty():
            frame1 = queue1.get()
            frame1_resized = cv2.resize(frame1, (320, 240))
            cv2.imshow("Camera 1", frame1_resized)

        if not queue2.empty():
            frame2 = queue2.get()
            frame2_resized = cv2.resize(frame2, (320, 240))
            cv2.imshow("Camera 2", frame2_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to exit
            break

        time.sleep(0.01)

    # Clean up
    cv2.destroyAllWindows()
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()
