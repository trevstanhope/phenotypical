import sys, cv2, os
from phenotypical import *

print os.getcwd()

img_dir = 'samples'
a = 'fig.jpg'
b = 'fig_subset.jpg'

training_path = os.path.join(img_dir, a)
sample_path = os.path.join(img_dir, b)

training_img = cv2.imread(training_path)
sample_img = cv2.imread(sample_path)
phenotype = Phenotype(name='test', samples=[training_img])
matcher = Matcher()
matcher.train(training_img, phenotype)
matcher.classify(sample_img)
