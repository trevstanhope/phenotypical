import sys, cv2, os
import fnmatch, re
from phenotypical import *

data_dir = 'data'
data_type = '.jpg'

print 'Generating matcher ...'
matcher = Matcher()

print 'Generating training dataset ...'
training_set = [
    'V1_a.jpg',
    'V1_b.jpg',
    'V1_c.jpg',
    'V1_d.jpg'
]
for f in training_set:
    if f.endswith(data_type):
        bgr = cv2.imread(os.path.join(data_dir, f))
        phenotype = f.split('_')[0] # phenotype is start of name before '_'
        ver = f.split('_')[1]
        obj = matcher.train(bgr, phenotype) 
        print phenotype, ver, obj

print 'Classifying test images ...'
test_set = [
    'V1_a3.jpg'
]
for f in test_set:
    bgr = cv2.imread(os.path.join(data_dir, f))
    res = matcher.classify(bgr)
    print f, res
