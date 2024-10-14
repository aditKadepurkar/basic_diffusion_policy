"""



"""

import cv2
import os

def create_recording_from_images(image_folder, video_path, fps=30):
    print("Saving video...", end="")
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(f"data/visual/avi/output_video{video_path}.avi", fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        # yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)  # Convert to YUV 4:2:0 format
        video.write(frame)
    
    video.release()
    cv2.destroyAllWindows()

    for image in images:
        os.remove(os.path.join(image_folder, image))
    
    print("Done")
