# coding=utf-8
import numpy as np

accuracy_list = []
with open("results.txt", "r", encoding="utf-8") as f:
    for line in f:
        accuracy = float(line.strip())
        accuracy_list.append(accuracy)

accuracy_mean = np.mean(accuracy_list)
accuracy_std = np.std(accuracy_list)

print("accuracy_list = {}".format(accuracy_list))
print("num_tests = {}".format(len(accuracy_list)))
print("accuracy: mean = {:.4f}\tstd = {:.4f}".format(accuracy_mean, accuracy_std))
