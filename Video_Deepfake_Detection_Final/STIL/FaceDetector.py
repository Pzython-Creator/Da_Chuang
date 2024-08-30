import cv2  
import torch  
import numpy as np  
import pickle  
import os
from facenet_pytorch import MTCNN  
  
class FaceDetector(object):  
    """  
    Face detector class  
    """  
  
    def __init__(self, mtcnn, video_path,video_name):  
        self.mtcnn = mtcnn  
        self.video_path = video_path  
        self.video_name=video_name
  
    def run(self, output_pkl_path):  
        """  
        Run the FaceDetector and save face bounding box information to a pkl file  
        """  
        cap = cv2.VideoCapture(self.video_path)  
        frame_count = 0  
        bounding_boxes = []  # To store bounding boxes for all frames  
  
        while True:  
            ret, frame = cap.read()  
            if not ret:  
                break  
  
            boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)  
            if boxes is not None:  
                bounding_boxes.append(boxes[0].tolist())  # Store bounding boxes for this frame  
  
            frame_count += 1  
  
        cap.release()  
        
  
        # Save bounding boxes to a pkl file  
        with open(output_pkl_path, 'wb') as f:  
            pickle.dump({self.video_name: bounding_boxes}, f)  

def find_video_file(directory, extensions):  
    """  
    在指定目录中寻找具有指定扩展名的视频文件，并返回第一个找到的文件名（包含完整路径）。  
  
    :param directory: 要搜索的文件夹路径  
    :param extensions: 要搜索的视频文件扩展名列表，如 ['.mp4', '.avi']  
    :return: 第一个找到的视频文件的完整路径，如果未找到则返回None  
    """  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if os.path.splitext(file)[1].lower() in extensions:  
                return os.path.join(root, file)  
    return None  
path=find_video_file('samples/manipulated_sequences/videos/c23/videos',['.mp4', '.avi'])
name = path.rsplit('.', 1)[0]  
# Run the app  
mtcnn = MTCNN(select_largest=False, post_process=False, device='cuda:0' if torch.cuda.is_available() else 'cpu')  
fcd = FaceDetector(mtcnn, path,name)  # Replace with your video file path  
fcd.run('samples/face_bbox.pkl')  # Replace with your desired output pkl file path