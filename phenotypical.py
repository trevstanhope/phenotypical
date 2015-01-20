import cv, cv2
import os

class BruteMatch:
    
    def __init__(self):
        self.SURF_HESSIAN_FILTER = 1000
        self.surf = cv2.SURF(self.SURF_HESSIAN_FILTER)
        self.matcher = cv2.BFMatcher()
        self.samples = {}
    
    def train(self, img_path, img_type):
        files = os.listdir(img_path)
        img_files = [s for s in files if img_type in s]
        for img in img_files:
            bgr = cv2.imread(os.path.join(img_path, img))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            (kp, des) = self.surf.detectAndCompute(gray, None)
            self.samples[img] = (bgr, kp, des)
                     
    def find_matches(self, bgr_1, MATCH_FACTOR=0.5, K=2):
        gray_1 = cv2.cvtColor(bgr_1, cv2.COLOR_BGR2GRAY)
        (kp_1, des_1) = self.surf.detectAndCompute(gray_1, None)
        res = {}
        for sample_name in self.samples:
            (bgr_2, kp_2, des_2) = self.samples[sample_name]
            matches = self.matcher.knnMatch(des_1, des_2, k=K)
            good = [m for (m,n) in matches if m.distance < MATCH_FACTOR * n.distance]
            res[sample_name] = len(good)
        return max(res, key=res.get)
        
if __name__ == '__main__':
    sample_path = 'tests/fig_subset.jpg'
    training_dir = 'samples'
    training_type = '.jpg'
    test = BruteMatch()
    test.train(training_dir, training_type)
    sample = cv2.imread(sample_path)
    res = test.find_matches(sample)
    
    cv2.waitKey(0)
    print res
    
