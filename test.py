from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import time
import json
import sys

def ResizeImage(image_path, size):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            resized_img = img.resize(size)
            resized_img.save(image_path, format="JPEG", quality=95)
    except Exception as e:
        print(f"An error occurred: {e}")
        return

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

def distance(histogram1, histogram2) :
    return np.sqrt(np.sum((histogram1 - histogram2) ** 2))

def CheckImage(image_path, row, col, positive, negative, bins=256) :
    ResizeImage(image_path, (224, 224))
    histograms= GenerateImageVectors(image_path, row, col, bins)
    diff1 = [ 0 for _ in range(row * col) ]
    diff2 = [ 0 for _ in range(row * col) ]
    for i in range(row * col):
        diff1[i] = distance(histograms[i], positive[i])
        diff2[i] = distance(histograms[i], negative[i])
    count_1 = 0
    count_2 = 0
    for i in range (row * col) :
        if diff1[i] < diff2[i] :
            count_1 += 1
        elif diff2[i] < diff1[i] :
            count_2 += 1
    if count_1 > count_2 :
        return 1, (count_1 * 100) / (row * col) 
    elif count_2 > count_1 :
        return 0, (count_2 * 100) / (row * col)
    else :
        return 2, 50

def GetDataFromJsonFile(filename) :
    with open(filename, 'r') as jsonfile:
        return json.load(jsonfile)
def GetAccuracy(directory_path, row, col, positive, negative, bins=256):
    accuracy = 0;
    count = 0
    path = directory_path + '\\positive'
    for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                count += 1
                result, prcg = CheckImage(file_path, row, col, positive, negative, bins)
                if result == 1:
                    accuracy += 1
                    print(f"{file_name} => [ positive: {prcg}%, negative : {100 - prcg}% ]")
                elif result == 0:
                    print(f"{file_name} => [ positive: {100 - prcg}%, negative : {prcg}% ]")
                else :
                    print(f"{file_name} => [ positive: {100 - prcg}%, negative : {prcg}% ]")
    path = directory_path + '\\negative'
    for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                count += 1
                result, prcg = CheckImage(file_path, row, col, positive, negative, bins)
                if result == 1:
                    print(f"{file_name} => [ positive: {prcg}%, negative : {100 - prcg}% ]")
                elif result == 0:
                    accuracy += 1
                    print(f"{file_name} => [ positive: {100 - prcg}%, negative : {prcg}% ]")
                else :
                    print(f"{file_name} => [ positive: {100 - prcg}%, negative : {prcg}% ]")
                       
    return (accuracy * 100) / count
    
print("Brain tumor detection using AI (testing) : ")
row = int(input("enter the number of rows : "))
col = int(input("enter the number of columns : "))
bins = int(input("enter the number of bins : ")) 
folder_path = input("enter the testing folder path : ")
histograms = GetDataFromJsonFile('trained.json')
positive = histograms['positive']
negative = histograms['negative']
print("accuracy : ", GetAccuracy(folder_path, row, col, positive, negative, bins), "%")
