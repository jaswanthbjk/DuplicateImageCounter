import os

import cv2.cv2 as cv2
import imagehash
from PIL import Image

from imaging_interview import compare_frames_change_detection
from imaging_interview import preprocess_image_change_detection


class CountDuplicates:
    def __init__(self, image_dir, metric):
        self.image_dir_path = image_dir
        self.image_paths = self.get_image_files()
        self.metric = metric

    def get_image_files(self):
        return [os.path.join(self.image_dir_path, image_name) for image_name
                in os.listdir(self.image_dir_path)]

    def duplicates_using_hash(self, hash_size=8):
        hashes = {}
        duplicates = []
        for image_path in self.image_paths:
            with Image.open(image_path) as img:
                temp_hash = imagehash.average_hash(img, hash_size)
                if temp_hash in hashes:
                    duplicates.append(image_path)
                else:
                    hashes[temp_hash] = image_path
        return duplicates

    def duplicates_using_absdiff(self, threshold):
        duplicates = []
        for count_1 in range(len(self.image_paths)):
            for count_2 in range(count_1 + 1, len(self.image_paths)):
                image1_path = self.image_paths[count_1]
                image2_path = self.image_paths[count_2]
                if image2_path in duplicates:
                    continue
                image1 = cv2.imread(image1_path)
                image2 = cv2.imread(image2_path)
                image1 = preprocess_image_change_detection(image1)
                image2 = preprocess_image_change_detection(image2)
                score, _, thresh = compare_frames_change_detection(
                    image1, image2, 150)
                if score < threshold:
                    duplicates.append(image2_path)
        return duplicates

    def get_duplicates(self):
        if self.metric == 'ahash':
            duplicates = self.duplicates_using_hash(hash_size=16)
        elif self.metric == 'abs_diff':
            duplicates = self.duplicates_using_absdiff(threshold=2750)
        else:
            print('Invalid Metric')
            return None
        return duplicates

    def __call__(self):
        duplicates = self.get_duplicates()
        return len(duplicates)


if __name__ == '__main__':
    counter = CountDuplicates('../ml-challenge/c23/', metric='ahash')
    print('Duplicates found with ahash technique is {}'.format(counter()))
    counter = CountDuplicates('../ml-challenge/c23/', metric='abs_diff')
    print('Duplicates found with absolute differnce is technique is {}'.format(
        counter()))
