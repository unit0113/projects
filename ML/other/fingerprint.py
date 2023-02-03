import os
import cv2
import random


# Get random sample
sample_file_path = r'C:\Users\unit0\Desktop\Python\projects\.venv\programs\NNs\Data\SOCOFing\Altered'
difficulty_folder = random.choice(os.listdir(sample_file_path))
sample_file = random.choice(os.listdir(os.path.join(sample_file_path, difficulty_folder)))
sample_full_path = os.path.join(sample_file_path, difficulty_folder, sample_file)

# Pull image, keypoints and descriptors from sample
sample = cv2.imread(sample_full_path)
sift = cv2.SIFT_create()
sample_kp, sample_descriptors = sift.detectAndCompute(sample, None)

# Loop through real fingerprints to find match
best_score = 0
filename = image = best_candidate_keypoints = best_mp = None
fingerprint_database_filepath = r'C:\Users\unit0\Desktop\Python\projects\.venv\programs\NNs\Data\SOCOFing\Real'
for file in os.listdir(fingerprint_database_filepath):
    fingerprint_image = cv2.imread(os.path.join(fingerprint_database_filepath, file))
    candidate_kp, candidate_descriptors = sift.detectAndCompute(fingerprint_image, None)

    # Get list of matches of descriptors
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {}).knnMatch(sample_descriptors, candidate_descriptors, k=2)
    match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]

    # Score and update best values
    keypoints = min(len(sample_kp), len(candidate_kp))
    score = len(match_points) / keypoints * 100
    if score > best_score:
        best_score = score
        filename = file
        image = fingerprint_image
        best_candidate_keypoints, best_mp = candidate_kp, match_points

# Show results
print(f'Sample File: {sample_file}')
print(f'Best Match: {filename}')
print(f'Score: {best_score}')
result = cv2.drawMatches(sample, sample_kp, image, best_candidate_keypoints, best_mp, None)
result = cv2.resize(result, None, fx=6, fy=6)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()