# PeakCET
A 2D peak detection algorithm called PeakCET based on Contour Edge Tracking and mass spectra vectors clustering for fast and effective analysis of two-dimensional (2D) fingerprints is presented. The algorithm starts by converting the raw file of comprehensive two-dimensional gas chromatographic (GC×GC) into a pixel matrix, followed by contour edge tracking. In order to optimize the detection results in the presence of false negatives, mass spectra vectors of local maximum points within the contour are subjected to Affinity Propagation (AP) clustering.
In order to facilitate readers to better observe the detection results, we visualize PeakCET as an interface.
# Requirements 
Python, version 3.9 or greater
OpenCV 3.4.2
Windows 11
Install additional libraries, listed in the requirements.txt
# Usage
1.	Prepare the file in mat format after data preprocessing.
2.	Create a folder for storing individual contour pictures.
3.	Run PeakCET.py, and the visualization interface will pop up.
4.	Click on the "input" in the menu bar and input the data file to visualize the data as a image, and then input visualization conditions. The results of contour detection can be obtained. 
5.	Click on the "Contours", "Standard-points" or "Each contour" button on the left to view the peak detection results. Click the button on the right to view individual contours one by one.
6.	Click on the "AP clustering" button, and a contour selection interface will pop up. Select the contour that you want to optimize, and the final optimized detection result will be obtained.
