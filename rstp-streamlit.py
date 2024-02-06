import cv2
from ultralytics import YOLO
import threading
from queue import Queue
import numpy as np
import  time
import streamlit as st
import utils
from concurrent.futures import ThreadPoolExecutor
import tracemalloc

tracemalloc.start()
RESIZE_DIMENSIONS = (640, 480)
MAX_QUEUE_SIZE = 10  # Set the maximum size of the queue

# OUTPUT VIDEOS ARE NO MORE WRITTEN OUT
# skip some frames with interval
# skip frames based on MSE
# Include concurrent.futures import ThreadPoolExecutor  to paralelize frames process
#

def run_tracker_in_thread_nowritting(filename, model, output_queue, video_output_path, camera_label):
    video = cv2.VideoCapture(filename)
    # Process video
    while True:
        ret, frame = video.read()  # Read the next frame

        if not ret:
            print(f"End of video stream from {camera_label}")
            break

        # Assuming each item in results is a detection object for a frame
        detections = model.track(frame, persist=True)
        res_plotted = detections[0].plot()
        output_queue.put(res_plotted)

        # Removed writing frames to output video

    # Release video capture when job is finished
    video.release()


def run_tracker_in_thread_frame_skipping(filename, model, output_queue, video_output_path, camera_label, skip_frames=20):
    video = cv2.VideoCapture(filename)

    frame_count = 0  # Counter to keep track of frames

    while True:
        ret, frame = video.read()

        if not ret:
            print(f"End of video stream from {camera_label}")
            break

        # Skip a certain number of frames to reduce the frame rate
        if frame_count % (skip_frames + 1) != 0:
            frame_count += 1
            continue

        # Process the frame
        detections = model.track(frame, persist=True)
        res_plotted = detections[0].plot()
        output_queue.put(res_plotted)

        frame_count += 1

    video.release()


def mse(imageA, imageB):
    # Compute the mean squared error between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def process_frame(frame, model):
    # Frame processing logic
    detections = model.track(frame, persist=True)
    res_plotted = detections[0].plot()
    return res_plotted


def run_tracker_in_thread_mse(filename, model, output_queue, video_output_path, camera_label, skip_rate=5, mse_threshold=1000):
    video = cv2.VideoCapture(filename)
    RESIZE_DIMENSIONS = (320, 240) # 854x480
    frame_count = 0
    ret, prev_frame_processed = video.read()  # First frame for initial comparison
    prev_frame_processed = cv2.resize(prev_frame_processed, RESIZE_DIMENSIONS)

    while True:
        ret, frame = video.read()

        if not ret:
            print(f"End of video stream from {camera_label}")
            break

        # Skip frames according to the specified rate
        if frame_count % skip_rate != 0:
            frame_count += 1
            continue

        frame = cv2.resize(frame, RESIZE_DIMENSIONS)


        # Calculate MSE between the current frame and the last processed frame
        frame_diff = mse(frame, prev_frame_processed)

        # Process the frame if the difference exceeds the threshold
        if frame_diff > mse_threshold:
            detections = model.track(frame, persist=True)
            res_plotted = detections[0].plot()
            output_queue.put(res_plotted)
            prev_frame_processed = frame.copy()

        frame_count += 1

    video.release()


def run_tracker_in_thread_localqueue(filename, model, output_queue, video_output_path, camera_label, reduced_width=320, reduced_height=240, skip_rate=5, mse_threshold=1000, max_queue_size=30):
    video = cv2.VideoCapture(filename)

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

        # Resize frame to reduce resolution
        frame = cv2.resize(frame, (reduced_width, reduced_height))

        # Skip frames according to the specified rate
        if frame_count % skip_rate != 0:
            frame_count += 1
            continue

        # Calculate MSE between the current frame and the last processed frame
        frame_diff = mse(frame, prev_frame_processed)

        # Check if the local queue is full
        if local_queue.full():
            local_queue.get()  # Discard the oldest frame

        # Add the new frame to the queue if it is sufficiently different
        if frame_diff > mse_threshold:
            local_queue.put(frame)

        # Process the frame from the queue
        if not local_queue.empty():
            frame_to_process = local_queue.get()
            detections = model.track(frame_to_process, persist=True)
            res_plotted = detections[0].plot()
            output_queue.put(res_plotted)
            prev_frame_processed = frame_to_process.copy()

        frame_count += 1

    video.release()


def run_tracker_in_thread_parallelized(filename, model, output_queue, video_output_path, camera_label, reduced_width=320, reduced_height=240, skip_rate=10, mse_threshold=1000, max_queue_size=30):
    video = cv2.VideoCapture(filename)
    executor = ThreadPoolExecutor(max_workers=2)  # Adjust based on your hardware

    frame_count = 0
    ret, prev_frame_processed = video.read()
    if ret:
        prev_frame_processed = cv2.resize(prev_frame_processed, (reduced_width, reduced_height))

    local_queue = Queue(maxsize=max_queue_size)

    while True:
        print(f"\n*From {camera_label}*")
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
    st.title("ThuliumX - TrackerV0")
    # Streamlit sidebar for parameter adjustments
    st.sidebar.title("Adjust Parameters")
    reduced_width = st.sidebar.slider("Reduced Width", min_value=100, max_value=1920, value=320, step=10)
    reduced_height = st.sidebar.slider("Reduced Height", min_value=100, max_value=1080, value=240, step=10)
    skip_rate = st.sidebar.slider("Skip Rate", min_value=1, max_value=30, value=10)
    mse_threshold = st.sidebar.slider("MSE Threshold", min_value=500, max_value=5000, value=1000)
    max_queue_size = st.sidebar.slider("Max Queue Size", min_value=10, max_value=100, value=30)

    #on = st.toggle('Activate feature')

    if 'is_processing' not in st.session_state:
        st.session_state['is_processing'] = False

    if st.sidebar.toggle('Activate Tracker' if not st.session_state['is_processing'] else 'Stop Processing'):
        st.session_state['is_processing'] = not st.session_state['is_processing']

    if st.session_state['is_processing']:

        # Initialize the YOLO model
        model_path = utils.DEFAULT_MODEL_PATH  # Update this path as needed

        # Initialize models and queues
        model1 = YOLO(model_path)
        model2 = YOLO(model_path)

        # Starting threads for reading camera stream (Producers)
        rtsp_url1 = utils.DEFAULT_RTSP_URL1
        rtsp_url2 = utils.DEFAULT_RTSP_URL2


        queue1 = Queue()
        queue2 = Queue()

        # Start threads for video processing
        #thread1 = threading.Thread(target=run_tracker_in_thread_parallelized, args=(rtsp_url1, model1, queue1, "output1.mp4", "Camera 2"), daemon=True)
        thread1 = threading.Thread(target=run_tracker_in_thread_parallelized, args=(rtsp_url1, model1, queue1, "output1.mp4", "Camera 2", reduced_width, reduced_height, skip_rate, mse_threshold, max_queue_size), daemon=True)
        #thread2 = threading.Thread(target=run_tracker_in_thread_parallelized, args=(rtsp_url2, model2, queue2, "output2.mp4", "Camera 1"), daemon=True)
        thread2 = threading.Thread(target=run_tracker_in_thread_parallelized, args=(rtsp_url2, model2, queue2, "output1.mp4", "Camera 1", reduced_width, reduced_height, skip_rate, mse_threshold, max_queue_size), daemon=True)
        thread1.start()
        thread2.start()

        # Streamlit loop for displaying frames
        col1, col2 = st.columns(2)
        with col1:
            st.header("Camera 1")
            frame_holder1 = st.empty()
        with col2:
            st.header("Camera 2")
            frame_holder2 = st.empty()

        while True:
            if not queue1.empty():
                frame1 = queue1.get()
                frame1_resized = cv2.resize(frame1, RESIZE_DIMENSIONS)
                frame_holder1.image(frame1_resized, channels="BGR")

            if not queue2.empty():
                frame2 = queue2.get()
                frame2_resized = cv2.resize(frame2, RESIZE_DIMENSIONS)
                frame_holder2.image(frame2_resized, channels="BGR")

            # Use Streamlit's way to handle loop interruption and waiting
            #if st.button("Stop"):
            #    break
            time.sleep(0.01)

        # Clean up
        thread1.join()
        thread2.join()

if __name__ == "__main__":
    main()
