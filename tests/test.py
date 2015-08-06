import sys, cv2, os
import fnmatch, re
from phenotypical import *

data_dir = 'data'
data_type = '.jpg'

print 'Generating matcher ...'
matcher = Matcher()

print 'Generating training dataset ...'
for file in os.listdir(data_dir):
    if file.endswith(data_type):
        bgr = cv2.imread(os.path.join(data_dir, file))
        phenotype = file.split('_')[0] # phenotype is start of name before '_'
        obj = matcher.train(bgr, phenotype) 
        print phenotype, obj

print 'Classifying test images ...'
res = matcher.classify(bgr)
print res
