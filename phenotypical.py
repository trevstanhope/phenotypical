"""
Phenotypical

MongoDB Phenotype (e.g. Soy V1)
"""
__author__ = 'Trevor Stanhope'
__version__ = 0.1

import cv2
import pymongo
import uuid
import sys

class Phenotype:
    def __init__(self, specie=None):
        self.specie = specie

class Matcher:
    def __init__(self, db=uuid.uuid4(), addr="127.0.0.1", port=27017, hessian=500):
        self.mongo_client = pymongo.MongoClient(addr, port)
        self.mongo_db = self.mongo_client[db]
        self.keypoint_filter = cv2.SURF(hessian, nOctaves=5, nOctaveLayers=3, extended=1, upright=1)
        self.matcher = cv2.BFMatcher()
    
    def train(self, bgr, phenotype):
        print phenotype.specie
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        (kp, des) = self.keypoint_filter.detectAndCompute(gray, None)
        return True
    
    def classify(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        (kp, des) = self.keypoint_filter.detectAndCompute(gray, None)
        matches = self.matcher.knnMatch(des, sample.des, k=K)
        good = [m for (m,n) in matches if m.distance < MATCH_FACTOR * n.distance]
        return phenotype

if __name__ == '__main__':
    training_path = sys.argv[1]
    sample_path = sys.argv[2]
    training_img = cv2.imread(training_path)
    sample_img = cv2.imread(sample_path)
    phenotype = Phenotype()
    matcher = Matcher()
    matcher.train(training_img, phenotype)
    matcher.classify(sample_img)
