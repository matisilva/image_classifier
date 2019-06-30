import os
import csv
CATEGORIES = range(16)
FILENAME = "/Users/silva/Downloads/train.csv"
TARGET_FOLDER = "/Users/silva/Documents/yolo_project/pytorch-yolo2/train_despegar"
FROM_FOLDER = "/Users/silva/Documents/yolo_project/pytorch-yolo2/train"
try:
    for x in range(CATEGORIES):
        os.mkdir("{}/{}".format(TARGET_FOLDER, str(x)))
except Exception:
    print("Ommiting dir creation")


with open(FILENAME) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in list(csvReader)[9:]:
        print(row)
        os.system("mv {}/{}.jpg {}/{}".format(FROM_FOLDER, row[1], TARGET_FOLDER, row[2]))