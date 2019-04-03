# Camera calibration with Tsai algorithm

## Input data format

The data used for the calibration is stored in the data.txt file. Each line is of the form :
```
X Y Z x y
```
Where (X, Y, Z) are the coordinates of the 3D points of the calibration rig and (x, y) their corresponding coordinates on the projection picture.

There can be as many lines as wanted.

## Compute the internal parameters of the camera

Just run main.py and will be computed :
- The projection matrix P, used to project 3D points of the world coordinates on the image.
- The calibration matrix K, from which can be deduced the parameters of the camera.
- The caracteristics of the pixels of the camera (base, height, angle).

## Resource

This work has been made following the subject.pdf file, proposed by the computer vision course of the ENPC school. See more details in it.
