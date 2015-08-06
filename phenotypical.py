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
import numpy as np

class Sample:
    def __init__(self, kp, des):
        self.kp = kp
        self.des = des
    def toDict(self):
        result = []
        for kp in self.kp:
            d = {
                'angle' : kp.angle,
                'octave' : kp.octave,
                'pt' : kp.pt,
                'size' : kp.size,
                'class_id' : kp.class_id,
                'response' : kp.response,
            }
            result.append(d)
        obj = {
            'kp' : result,
            'des' : self.des.tolist()
        }
        return obj
    def fromDict(d):
        self.kp = d['kp']
        self.des = d['des']
        return self

class Matcher:
    def __init__(self, db=uuid.uuid4().hex, addr="127.0.0.1", port=27017, hessian=500):
        self.mongo_client = pymongo.MongoClient(addr, port)
        self.mongo_db = self.mongo_client[db]
        self.keypoint_filter = cv2.SURF(hessian, nOctaves=5, nOctaveLayers=3, extended=1, upright=1)
        self.matcher = cv2.BFMatcher()
    def train(self, bgr, phenotype):
        """
        inserts sample image into a mongoDB collection of the named phenotype
        Each training image is used to generate a document which contains SURF and phenotype information
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        (keypoints, descriptors) = self.keypoint_filter.detectAndCompute(gray, None)
        collection = self.mongo_db[phenotype]
        sample = Sample(keypoints, descriptors)
        _id = collection.insert(sample.toDict())
        return _id # prints the object ID of the new sample image
    def classify(self, bgr, N=2, alpha=0.5):
        """
        Classify an input BGR image as a phenotype
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        (kp, des) = self.keypoint_filter.detectAndCompute(gray, None)
        results = []
        for phenotype in self.mongo_db.collection_names():
            if not phenotype == 'system.indexes':
                for sample in self.mongo_db[phenotype].find():
                    des2 = np.array(sample['des'], np.float32) # to cast to Float32
                    matches = self.matcher.knnMatch(des, des2, k=N)
                    good = [m for (m,n) in matches if m.distance < alpha * n.distance]
                    results.append((phenotype, sample['_id'], len(good)))
        descending = sorted(results, key=lambda x: x[-1], reverse=True)
        return descending[0] # return doc info of best match
