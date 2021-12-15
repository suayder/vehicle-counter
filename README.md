# vehicle-counter

Object Tracker to counter vehicles in video.

Basicaly we have 5 videos, each video have a .py file to this with the same name. In essence all the files go through the same logic it differs only on parameter setting.

Follow the description of the files:

- `road_traffic_{n}.py` - corresponding code to the video processing
- `motociata.py` - Bolsonaro's motociata in São Paulo Brazil. Counter of motorcycles
- `tracker.py` - tracker by centroid proximity
- `counter.py` - object counter -> get the object and its history from the tracker, if an object is in some ammount of consecutive frames it will be counted. See in the code the parameters.
- `*.png` - you can see just the screenshot of each count in each video.

> This code was built in order to attend the second test at the laboratory of computer vision discipline on the IME-USP