import csv
import pickle
import torch
# 打开文件
# with open('/home/ubuntu/qr/CAMELYON16/label.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         if("patient" not in row[0]):
#             print(row)
path = "/home/ubuntu/qr/CAMELYON16/pt/normal_042.pt"  # 文件路径
#with open(path, "rb") as file:
    #data = pickle.load(file)
data=torch.load(path)
print(data.shape)