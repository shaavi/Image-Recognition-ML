#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# get the reference to the webcam
CAMERA = cv2.VideoCapture(0)

while(True):
    # read a new frame
    _, frame = CAMERA.read()

    # show the frame
    cv2.imshow("Capturing frames", frame)

    # quit camera if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

CAMERA.release()
cv2.destroyAllWindows()


# In[3]:


# get the reference to the webcam
CAMERA = cv2.VideoCapture(0)
HEIGHT = 500

while(True):
    # read a new frame
    _, frame = CAMERA.read()
    
    # flip the frame
    frame = cv2.flip(frame, 1)

    # rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * HEIGHT) # landscape orientation - wide image
    frame = cv2.resize(frame, (res, HEIGHT))

    # add rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (0, 255, 0), 2)

    # show the frame
    cv2.imshow("Capturing frames", frame)

    # quit camera if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

CAMERA.release()
cv2.destroyAllWindows()


# In[4]:


# get the reference to the webcam
CAMERA = cv2.VideoCapture(0)
HEIGHT = 500
RAW_FRAMES = []

while(True):
    # read a new frame
    _, frame = CAMERA.read()
    
    # flip the frame
    frame = cv2.flip(frame, 1)

    # rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * HEIGHT) # landscape orientation - wide image
    frame = cv2.resize(frame, (res, HEIGHT))

    # add rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (0, 255, 0), 2)

    # show the frame
    cv2.imshow("Capturing frames", frame)

    key = cv2.waitKey(1)

    # quit camera if 'q' key is pressed
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("s"):
        # save the frame
        RAW_FRAMES.append(frame)
        
        # preview the frame
        plt.imshow(frame)
        plt.show()

CAMERA.release()
cv2.destroyAllWindows()


# In[5]:


# show raw frames
for frame in RAW_FRAMES:
    plt.imshow(frame)
    plt.show()


# In[6]:


IMAGES = []

for frame in RAW_FRAMES:
    # get ROI
    roi = frame[75+1:425-1, 300+1:650-1]
    
    # parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    IMAGES.append(roi)
    
    plt.imshow(roi)
    plt.show()


# In[7]:


len(IMAGES)


# In[8]:


IMAGES[0].shape


# In[ ]:




