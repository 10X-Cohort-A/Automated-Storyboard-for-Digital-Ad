from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
from pathlib import Path
import cv2
import os

class ObjectDetection:
    def __init__(self) -> None:
        """
        Initialize ObjectDetection class with YOLOv3 model pretrained on VOC dataset.
        """
        try:
            self.net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
            print("Object detection model created")
        except Exception as e:
            print(f"Error while creating object detection: {e}")
        
        
    def detect_from_image(self, image_path: str) -> tuple:
        """
        Perform object detection on a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            tuple: Tuple containing detected class IDs, scores, bounding boxes, and the image.
        """
        try:
            x, img = data.transforms.presets.yolo.load_test(image_path, short=512)
            class_IDs, scores, bounding_boxs = self.net(x)
            print("Object detected from image successfully")
            return class_IDs, scores, bounding_boxs, img
        except Exception as e:
            print(f"Error while detecting objects from image: {e}")
            return (), (), (), None
    
    def detect_objects_and_info(self, image_path: str) -> list:
        """
        Detect objects in an image and return information about each detected object.

        Args:
            image_path (str): Path to the image file.

        Returns:
            list: List of dictionaries containing information about each detected object.
        """
        try:
            class_IDs, scores, bounding_boxes, img = self.detect_from_image(image_path)
            
            if img is None:
                return []

            # Process detected objects
            detected_objects = []
            index = 0
            for class_id, score, bbox in zip(class_IDs[0], scores[0], bounding_boxes[0]):

                cid = int(class_id[0][0].asnumpy())
                class_name = self.net.classes[cid]

                score = float(score.asscalar())

                x_min, y_min, x_max, y_max = [int(coord.asscalar()) for coord in bbox]

                width = x_max - x_min
                height = y_max - y_min
                starting_position = (x_min, y_min)

                if score > 0.5:
                    detected_objects.append({
                        "class_name": class_name,
                        "width": width,
                        "height": height,
                        "starting_position": starting_position
                    })

                index += 1

            return detected_objects
        except Exception as e:
            print(f"Error while detecting objects and info: {e}")
            return []
    
    def detect_from_video(self, video_path: str) -> dict:
        """
        Perform object detection on a video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            dict: Dictionary containing detected class IDs, scores, bounding boxes, and images for each frame.
        """
        try:
            result = {'class_IDs':[], 'scores':[], 'bounding_boxs':[], 'img':[]}
            
            cap = cv2.VideoCapture(video_path)
            i = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret is False:
                    break
              
                cv2.imwrite('frame'+str(i)+'.jpg', frame)
                x, img = data.transforms.presets.yolo.load_test('frame'+str(i)+'.jpg', short=512)
                class_IDs, scores, bounding_boxs = self.net(x)
                
                result['class_IDs'].append(class_IDs)
                result['scores'].append(scores)
                result['bounding_boxs'].append(bounding_boxs) 
                result['img'].append(img)
                if os.path.isfile('frame'+str(i)+'.jpg'):
                    os.remove('frame'+str(i)+'.jpg')
                i += 1
            cap.release()
            cv2.destroyAllWindows()
            
            print("Object detection completed on video successfully")
            return result
        
        except Exception as e:
            print(f"Error while detecting objects from video: {e}")
            return {}
        
    def plot_detection(self, img, class_IDs, scores, bounding_boxs) -> None:
        """
        Plot the object detection results on the image.

        Args:
            img: Image on which detection results will be plotted.
            class_IDs: IDs of detected classes.
            scores: Confidence scores of detected objects.
            bounding_boxs: Bounding boxes of detected objects.
        """
        try:
            utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                                class_IDs[0], class_names=self.net.classes)
            plt.show() 
            print("Object detection plot generated successfully")
        
        except Exception as e:
            print(f"Error while plotting object detection: {e}")
