# EX04 IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the necessary libraries and read the original image and save it as a image variable.



### Step2:
Translate the image using a function warpPerpective()



### Step3:
Scale the image by multiplying the rows and columns with a float value.



### Step4:
Shear the image in both the rows and columns.



### Step5:
Find the reflection of the image.



## Program:
```
Developed By:JANARTHANAN V K 
Register Number: 212222230051
```
i)Image Translation

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

input_image=cv2.imread("logo.png")
input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M=np.float32([[1,0,50],  [0,1,100],  [0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(cols,rows))
plt.axis('off')
plt.imshow(translated_image)
plt.show()
```
ii) Image Scaling
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("logo.png")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
M = np.float32([[1.5,0,0],[0,1.7,0],[0,0,1]])
scaled_img = cv2.warpPerspective(org_image,M,(cols*2,rows*2))
plt.imshow(org_image)
plt.show()
```


iii)Image shearing
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("logo.png")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
M_X = np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
M_Y = np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
sheared_img_xaxis = cv2.warpPerspective(org_image,M_X,(int(cols*1.5),int(rows*1.5)))
sheared_img_yaxis = cv2.warpPerspective(org_image,M_Y,(int(cols*1.5),int(rows*1.5)))
plt.imshow(sheared_img_xaxis)
plt.show()
plt.imshow(sheared_img_yaxis)
plt.show()
```


iv)Image Reflection
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("logo.png")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
M_X = np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
M_Y = np.float32([[-1,0,cols],[0,1,0],[0,0,1]])
reflected_img_xaxis = cv2.warpPerspective(org_image,M_X,(int(cols),int(rows)))
reflected_img_yaxis = cv2.warpPerspective(org_image,M_Y,(int(cols),int(rows)))
plt.imshow(reflected_img_xaxis)
plt.show()
plt.imshow(reflected_img_yaxis)
plt.show()

```



v)Image Rotation
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
in_img=cv2.imread("photograph.jpg")
in_img=cv2.cvtColor(in_img,cv2.COLOR_BGR2RGB)
rows,cols,dim=in_img.shape
angle=np.radians(10)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],
              [np.sin(angle),np.cos(angle),0],
              [0,0,1]])
rotated_img=cv2.warpPerspective(in_img,M,(int(cols),int(rows)))
plt.axis('off')
plt.imshow(rotated_img)
plt.show()
```



vi)Image Cropping

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
org_image = cv2.imread("logo.png")
org_image = cv2.cvtColor(org_image,cv2.COLOR_BGR2RGB)
plt.imshow(org_image)
plt.show()
rows,cols,dim = org_image.shape
cropped_img=org_image[80:900,80:500]
plt.imshow(cropped_img)
plt.show()


```
## Output:

### Input Image
<img src="https://github.com/Janarthanan2/DIP_EX04_IMAGE-TRANSFORMATIONS/assets/119393515/725586f3-8a09-4d45-b66c-5d7cace62a22">

### i)Image Translation

<img src="https://github.com/Janarthanan2/DIP_EX04_IMAGE-TRANSFORMATIONS/assets/119393515/b8a95a67-2855-433d-8893-55d198cbb953">


<br>

### ii) Image Scaling

<img src="https://github.com/Janarthanan2/DIP_EX04_IMAGE-TRANSFORMATIONS/assets/119393515/695e9a6e-7bd5-4352-9af5-08260a68e1b0">
<br>




### iii)Image shearing

<img src="https://github.com/Janarthanan2/DIP_EX04_IMAGE-TRANSFORMATIONS/assets/119393515/ece1d988-3671-4496-952e-091927857a94">
<img src="https://github.com/Janarthanan2/DIP_EX04_IMAGE-TRANSFORMATIONS/assets/119393515/b77abc31-453e-46bf-ac04-637898b521f4">
<br>


### iv)Image Reflection

<img src="https://github.com/Janarthanan2/DIP_EX04_IMAGE-TRANSFORMATIONS/assets/119393515/ca88135e-4912-40b4-b6cb-fa6eb7bfbcad">
<img src="https://github.com/Janarthanan2/DIP_EX04_IMAGE-TRANSFORMATIONS/assets/119393515/cd401135-5ebf-4474-a912-59db75b1f53f">

<br>


### v)Image Rotation
<img src="https://github.com/Janarthanan2/DIP_EX04_IMAGE-TRANSFORMATIONS/assets/119393515/bfab3d13-37d2-438f-8e56-c3ec54abaf4b">
<br>



### vi)Image Cropping
<img src="https://github.com/Janarthanan2/DIP_EX04_IMAGE-TRANSFORMATIONS/assets/119393515/06abaf12-0287-4ae4-8eed-ff7b872e4c7b">
<br>




## Result: 
Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
