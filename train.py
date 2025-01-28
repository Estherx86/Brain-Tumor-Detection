from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import time
import csv
import json

def ResizeImage(image_path, size):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            resized_img = img.resize(size)
            resized_img.save(image_path, format="JPEG", quality=95)
    except Exception as e:
        print(f"An error occurred: {e}")

def GenerateImageVectors(image_path, row, column, bins=256):
    img = Image.open(image_path).convert("L")
    width, height = img.size

    step_x = width // column
    step_y = height // row
    
    boxes = [
        (i * step_x, j * step_y, (i + 1) * step_x, (j + 1) * step_y)
        for j in range(row) for i in range(col)
    ]
    
    histograms = []
    for box in boxes:
        cropped_img = np.array(img.crop(box))
        histogram = np.zeros(bins)
        for element in cropped_img.flatten():
            if element >= 0 and element < bins :
                histogram[element] += 1
        histograms.append(histogram)
    return histograms

def SaveToJson(filename, data):
    with open(filename, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)
        
def GetBestHistogramCombination(filename, directory_path, row, col, bins=256):
    progress = 0
    files = os.listdir(directory_path)
    all = []
    for file_name, prg in zip(files, tqdm(range(len(files)), desc="Preparing Histograms")):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            ResizeImage(file_path, (224,224))
            all.append(GenerateImageVectors(file_path, row, col, bins))
    SaveToJson(filename, { 'features': np.array(all).tolist()});        
    optimal_combination = [ np.zeros(bins) for _ in range(row * col) ]
    for i in tqdm(range(row * col), desc="Training") :
        for j in range(bins) :
            sum = 0
            for k in range(len(all)):
                sum += all[k][i][j]
            optimal_combination[i][j] = sum / len(all)
            
    return optimal_combination

print("Brain tumor detection using AI (training): ")
row = int(input("enter the number of rows : "))
col = int(input("enter the number of columns : "))
bins = int(input("enter the number of bins : ")) 
tumor_path = input("enter the tumor folder path : ")
notumor_path = input("enter the notumor folder path : ")
positive = GetBestHistogramCombination('positive.json', tumor_path, row, col, bins)
negative = GetBestHistogramCombination('negative.json', notumor_path, row, col, bins)
SaveToJson('trained.json', { 'positive': np.array(positive).tolist(), 'negative': np.array(negative).tolist() })
