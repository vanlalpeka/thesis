import csv 
import time 
import json
import os
import datetime
import sys

with open("params/img.csv") as f:
    params = csv.DictReader(f, delimiter=';')
    i = 1
    row = int(sys.argv[1])
    for param in params:
        if i == row:
            print(param)
        i = i+1

    # params=csv.reader(f)


