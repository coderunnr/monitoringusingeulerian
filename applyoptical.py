# import numpy as np
# import cv2

# cap =cv2.VideoCapture(0)

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/liverecord.avi',fourcc, cap.get(cv2.CAP_PROP_FPS), (640,480))

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret==True:
        

#         # write the flipped frame
#         out.write(frame)

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# import numpy as np
# import cv2
# cap = cv2.VideoCapture('/home/kanishk/programming/eulerian-magnification-master/baby_laplacian.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/output.avi',fourcc,cap.get(cv2.CAP_PROP_FPS), (280,160))
# # params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# # Create some random colors
# color = np.random.randint(0,255,(100,3))
# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
# while(1):
#     ret,frame = cap.read()
#     if ret==True:
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # calculate optical flow
#         p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#         # Select good points
#         good_new = p1[st==1]
#         good_old = p0[st==1]
#         # draw the tracks
#         for i,(new,old) in enumerate(zip(good_new,good_old)):
#             a,b = new.ravel()
#             c,d = old.ravel()
#             mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#             frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
#         img = cv2.add(frame,mask)
#         cv2.imshow('frame',img)
#         out.write(img)
#         k = cv2.waitKey(30) & 0xff
#         if k == 27:
#             break
#         # Now update the previous frame and previous points
#         old_gray = frame_gray.copy()
#         p0 = good_new.reshape(-1,1,2)
#     else:
#         break    
# cv2.destroyAllWindows()
# out.release()
# cap.release()




import numpy as np
import cv2
cap = cv2.VideoCapture('/home/kanishk/programming/eulerian-magnification-master/showcase/output_live__laplacian_levels4_min0.4_max0.8_amp10.avi')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/kanishk/programming/eulerian-magnification-master/showcase/output_live_project_demo.avi',fourcc,cap.get(cv2.CAP_PROP_FPS), (640,480))
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
try:
    while(1):
        ret,frame = cap.read()
        if ret==True:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)
            cv2.imshow('frame',img)
            out.write(img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break         
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
 
finally:
    out.release()
    cv2.destroyAllWindows()
    cap.release()