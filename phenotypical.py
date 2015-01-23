import cv, cv2
import os

class Sample:
    def __init__(self, bgr, kp, des):
        self.bgr = bgr
        self.kp = kp
        self.des = des

class BruteMatch:
    
    def __init__(self):
        print 'init'
        self.SURF_HESSIAN_FILTER = 1000
        self.surf = cv2.SURF(self.SURF_HESSIAN_FILTER)
        self.matcher = cv2.BFMatcher()
        self.samples = {}
    
    def train(self, img_path, img_type):
        print 'train'
        files = os.listdir(img_path) # OS dependent
        img_files = [s for s in files if img_type in s]
        for img in img_files:
            print 'training %s' % img
            bgr = cv2.imread(os.path.join(img_path, img))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            (kp, des) = self.surf.detectAndCompute(gray, None)
            self.samples[img] = Sample(bgr, kp, des)
                     
    def find_matches(self, bgr, MATCH_FACTOR=0.5, K=2):
        print 'find match'
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        (kp, des) = self.surf.detectAndCompute(gray, None)
        res = {}
        print self.samples
        for sample_name in self.samples:
            sample = self.samples[sample_name]
            matches = self.matcher.knnMatch(des, sample.des, k=K)
            good = [m for (m,n) in matches if m.distance < MATCH_FACTOR * n.distance]
            res[sample_name] = len(good)
        return res
                
    def show_images(self, img):
        sample_name = self.find_match(img)
        sample = self.samples[sample_name]
        sample_img = sample.bgr
        
    def find_best(self, res):
        return max(res, key=res.get) # not very statistical
        
if __name__ == '__main__':
    new_img_path = 'tests/fig_subset_rotated.jpg'
    cam = cv2.VideoCapture()
    (s, bgr) = cam.read()
    training_dir = 'samples'
    training_type = '.jpg'
    test = BruteMatch()
    test.train(training_dir, training_type)
    
    #new_img = cv2.imread(new_img_path)
    cam = cv2.VideoCapture()
    print 's'
    while True:
        (s, new_img) = cam.read()
        if s:
            print 's'
            res = test.find_matches(new_img)
            best = test.find_best(res)
            print best
        else:
            print 'f'
