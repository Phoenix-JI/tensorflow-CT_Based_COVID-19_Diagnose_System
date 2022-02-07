# CT-Based-COVID-19-Diagnose-by-Image-Processing-and-Deep-Learning
### This is my undergraduate honour project in Hong Kong Baptist University, and thanks to my supervisor Prof. CHU Xiaowen
This project proposed the deep learning and image processing method to undertake the diagnosis on 2D CT image and 3D CT volume. The first proposed deep learning method applied 3D-UNet model to undertake the segmentation of the lung area from the raw 3D CT volume, and the image processing method can also do this part. And the second deep learning method offers the 3D CNN model to classify the COVID-19 cases normal CT volume followed by another 3D CNN model to classify the COVID-19 cases into three categories according to the infectious area. Then, the last 3D U-Net model could segment the infectious area from the confirmed cases and the slice with largest infectious area would be selected. Finally, the diagnosis report will be generated based on the previous output. The similar operation can be conducted on the 2D CT image.
## System Architecture
![This is an image](https://github.com/Phoenix-JI/CT-Based-COVID-19-Diagnose-by-Image-Processing-and-Deep-Learning/blob/main/System%20Architecture.png)
## Diagnose Result
![This is an image](https://github.com/Phoenix-JI/CT-Based-COVID-19-Diagnose-by-Image-Processing-and-Deep-Learning/blob/main/Diagnose%20Result.png)
