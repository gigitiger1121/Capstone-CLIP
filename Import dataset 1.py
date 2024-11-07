import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

CATEGORIES = ["Au", "CASIA 2 Groundtruth", "Tp"]
Authentic = []
Groundtruth = []
Tampered = []
DATADIR = path = "C:/Users/gigi1/CLIP/dataset/CASIA2/Au"

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    listDir = os.listdir(path)
    listDir.sort()
    if  category == "Au":
        for img in listDir:
            try:
                image = cv2.imread(os.path.join(path, img))
                if image is not None:
                    Authentic.append(image)
            except Exception as e:
                print(str(e))
    elif category == "CASIA 2 Groundtruth":
        for img in listDir:
            try:
                image = cv2.imread(os.path.join(path, img))
                if image is not None:
                    Groundtruth.append(image)
            except Exception as e:
                print(str(e))
    else:
        for img in listDir:
            try:
                image = cv2.imread(os.path.join(path, img))
                if image is not None:
                    Tampered.append(image)
            except Exception as e:
                print(str(e))
