
# SOURCE_VIDEOS_DIR = 'eulerian_source_videos'
# DESTINATION_VIDEOS_DIR = 'showcase'
# def apply_eulerian(source_filename,i):
#     image_processing = 'laplacian'
#     pyramid_levels = 4

#     name='output_live'
#     amplification_factor=10
#     cutoff=16
#     lower_hertz=0.4
#     upper_hertz=0.8
#     framerate=30

#     # source_filename = name + '.avi'

#     source_path = os.path.join(SOURCE_VIDEOS_DIR, source_filename)
#     vid, fps = load_video_float(source_path)

#     vid = em.eulerian_magnification(
#         vid, fps,
#         freq_min=lower_hertz,
#         freq_max=upper_hertz,
#         amplification=amplification_factor,
#         pyramid_levels=pyramid_levels
#     )
#     source_path = os.path.join(DESTINATION_VIDEOS_DIR, source_filename)
#     file_name = os.path.splitext(source_path)[0]
#     file_name = file_name + "__" + image_processing + "_levels" + str(pyramid_levels) + "_min" + str(
#         lower_hertz) + "_max" + str(upper_hertz) + "_amp" + str(amplification_factor) +"number" + str(i)
#     save_video(vid, fps, file_name + '.avi')




# import numpy as np
# import cv2
# import os
# import requests
# import eulerian_magnification as em
# from eulerian_magnification.io import save_video,load_video_float

# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/eulerian_source_videos/output_live_real.avi',fourcc,cap.get(cv2.CAP_PROP_FPS), (640,480))
# i=0
# # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # vid_frames = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), height, width, 3), dtype='uint8')
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
    


#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     # if i%5==0:
#     # 	cv2.imwrite("/home/kanishk/programming/eulerian-magnification-master/imageslive/live"+str(i)+".jpg",gray)
#     # 	print('Done')
#     cv2.imshow('frame',frame)
#     out.write(frame)
    
    
#     if i%199==0:
#         fps=int(cap.get(cv2.CAP_PROP_FPS))
#         out.release()
#         apply_eulerian('output_live_real.avi',i)
#         out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/eulerian_source_videos/output_live_real.avi',fourcc,cap.get(cv2.CAP_PROP_FPS), (640,480))
#         i=i+1
#         continue    
#     else:
#         i=i+1    

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release        
# cv2.destroyAllWindows()
# When everything done, release the capture






# import numpy as np
# import cv2

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/output_doll.avi',fourcc,30, (640,480))

# i=0
# while(i<300):
#     img=cv2.imread('/home/kanishk/Downloads/doll_test.jpg')
#     img=cv2.resize(img, (640, 480))
#     out.write(img)
#     i=i+1 


# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('/home/kanishk/programming/eulerian-magnification-master/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/eulerian_source_videos/output_live.avi',fourcc,cap.get(cv2.CAP_PROP_FPS), (640,480))


i=0
a=0
b=0
c=0
d=0
while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
  



    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_gray=gray
    roi_color=img
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x-int(w/2),y+h),(x+w+int(w/2),y+h+h+h),(255,0,0),2)
        
        if i < 5:
            a=x-int(w/2)
            b=y+h
            c=x+w+int(w/2)
            d=y+h+h+h
            i=i+1
        else:    
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[b:d,a:c]
            cv2.imshow('img',img)
            roi_color=cv2.resize(roi_color, (640, 480))
            out.write(roi_color)


        

        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # if i==5:        
    #     out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/output_live.avi',fourcc,cap.get(cv2.CAP_PROP_FPS), (d-b,c-a))


  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()