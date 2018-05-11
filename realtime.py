
SOURCE_VIDEOS_DIR = 'eulerian_source_videos'
DESTINATION_VIDEOS_DIR = 'showcase'
def apply_eulerian(source_filename,i):
    image_processing = 'laplacian'
    pyramid_levels = 4

    name='output_live'
    amplification_factor=10
    cutoff=16
    lower_hertz=0.4
    upper_hertz=0.8
    framerate=30

    # source_filename = name + '.avi'

    source_path = os.path.join(SOURCE_VIDEOS_DIR, source_filename)
    vid, fps = load_video_float(source_path)

    vid = em.eulerian_magnification(
        vid, fps,
        freq_min=lower_hertz,
        freq_max=upper_hertz,
        amplification=amplification_factor,
        pyramid_levels=pyramid_levels
    )
    source_path = os.path.join(DESTINATION_VIDEOS_DIR, source_filename)
    file_name = os.path.splitext(source_path)[0]
    file_name = file_name + "__" + image_processing + "_levels" + str(pyramid_levels) + "_min" + str(
        lower_hertz) + "_max" + str(upper_hertz) + "_amp" + str(amplification_factor) +"number" + str(i)
    save_video(vid, fps, file_name + '.avi')




import numpy as np
import cv2
import os
import requests
import eulerian_magnification as em
from eulerian_magnification.io import save_video,load_video_float

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/eulerian_source_videos/output_live_real.avi',fourcc,cap.get(cv2.CAP_PROP_FPS), (640,480))
i=0
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# vid_frames = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), height, width, 3), dtype='uint8')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    


    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # if i%5==0:
    # 	cv2.imwrite("/home/kanishk/programming/eulerian-magnification-master/imageslive/live"+str(i)+".jpg",gray)
    # 	print('Done')
    cv2.imshow('frame',frame)
    out.write(frame)
    
    
    if i%299==0:
        fps=int(cap.get(cv2.CAP_PROP_FPS))
        out.release()
        apply_eulerian('output_live_real.avi',i)
        out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/eulerian_source_videos/output_live_real.avi',fourcc,cap.get(cv2.CAP_PROP_FPS), (640,480))
        i=i+1
        continue    
    else:
        i=i+1    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release        
cv2.destroyAllWindows()