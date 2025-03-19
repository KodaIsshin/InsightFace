import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import insightface
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps
import random

df = pd.read_csv("modified.csv")
missing_files = df[~df["NewPath"].apply(os.path.exists)]
if not missing_files.empty:
    print("Missing files:")
    print(missing_files)