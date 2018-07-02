import os

model_values = open('model_values.h', 'w')
model_values.write('#pragma once')
model_values.write('static const float model_weight_Pnet_[]= {')

with open('Pnet.txt', 'r') as f:
    lines = f.readlines()
    new_lines = [float(l[1: len(l)-2]) for l in lines]
    idx =1
    for ll in new_lines:
        if idx == len(new_lines):
            model_values.write(str(ll))
        else:
            model_values.write(str(ll)+',')
        if idx % 10 == 0:
            model_values.write('\n')
        idx = idx + 1
    model_values.write('};\n')

model_values.write('static const float model_weight_Rnet_[]= {')
with open('Rnet.txt', 'r') as f:
    lines = f.readlines()
    new_lines = [float(l[1: len(l)-2]) for l in lines]
    idx =1
    for ll in new_lines:
        if idx == len(new_lines):
            model_values.write(str(ll))
        else:
            model_values.write(str(ll)+',')
        if idx % 10 == 0:
            model_values.write('\n')
        idx = idx + 1
    model_values.write('};\n')

model_values.write('static const float model_weight_Onet_[]= {')
with open('Onet.txt', 'r') as f:
    lines = f.readlines()
    new_lines = [float(l[1: len(l)-2]) for l in lines]
    idx =1
    for ll in new_lines:
        if idx == len(new_lines):
            model_values.write(str(ll))
        else:
            model_values.write(str(ll)+',')
        if idx % 10 == 0:
            model_values.write('\n')
        idx = idx + 1
    model_values.write('};\n')