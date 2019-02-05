#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import cv2


# In[3]:


color_1 = [255, 0, 0]      # red
color_2 = [0, 255, 0]      # green
color_3 = [0, 0, 255]      # blue
color_4 = [127, 127, 127]  # grey
plt.imshow(np.array([
    [color_1, color_2],
    [color_3, color_4],
]))


# In[4]:


colors = [
    [
        [0, 0, 255], # blue
        [0, 255, 0] # green
    ],
    [
        [255, 0, 0], # red
        [255, 255, 0] # yellow
    ]
]

print(np.array(colors).shape)

plt.imshow(colors)


# In[8]:


colors = [
    [
        [0, 0, 255], # blue
        [0, 255, 0]  # green
    ],
    [
        [255, 0, 0],  # red
        [255, 255, 0] # yellow
    ]
]

print(np.array(colors).shape)

plt.imshow(colors)
start_row = 0
for row in colors:
    start_col = -0.25
    for color in row:
        plt.text(start_col, start_row, str(color))
        start_col += 1
    start_row += 1


# In[15]:


SIZE = 4
# SIZE = 10
# SIZE = 4000

colors = np.array(
    np.array([
        np.array([np.random.randint(0, 255, 3) for x in range(SIZE)]) for x in range(SIZE)
    ])
)

print(np.array(colors).shape)

plt.imshow(colors)


# In[34]:


# using openCV
# read image
image = cv2.imread("C:/Users/user/Downloads/cupcakes/cupcka.jpg")


# In[21]:


type(image)


# In[22]:


image.shape


# In[37]:


plt.imshow(image)


# In[38]:


# parse BRG to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show image
plt.imshow(image)


# In[39]:


# parse image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show image
plt.imshow(gray, cmap='gray')


# In[40]:


WIDTH = 300
HEIGHT = 300

# resize, ignoring aspect ratio
resized = cv2.resize(image, (WIDTH, HEIGHT))

# show image
plt.imshow(resized)


# In[41]:


aspect = image.shape[1] / float(image.shape[0])
print(image.shape[1])
print(image.shape[0])
print(aspect)

if(aspect > 1):
    # landscape orientation - wide image
    res = int(aspect * HEIGHT)
    scaled = cv2.resize(image, (res, HEIGHT))
if(aspect < 1):
    # portrait orientation - tall image
    res = int(WIDTH / aspect)
    scaled = cv2.resize(image, (WIDTH, res))
if(aspect == 1):
    scaled = cv2.resize(image, (WIDTH, HEIGHT))

# show image
plt.imshow(scaled)


# In[43]:


def crop_center(img, cropx, cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy, startx:startx+cropx]
    # yes, the function above should match resize and take a tuple...

# Scaled image
cropped = crop_center(scaled, WIDTH, WIDTH)

# show image
plt.imshow(cropped, cmap='gray')


# In[51]:


image = cropped.copy()

# add text
cv2.putText(image, "Write Something!", (70, 270), 
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 240, 150), 2)

# add line
cv2.line(image, (75, 280), (280, 280), (50, 100, 250), 3)

# show image
plt.imshow(image)


# In[ ]:




