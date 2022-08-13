import numpy as np

from HaarLikeFeature import HaarLikeFeature
from HaarLikeFeature import feature_types
from integralimage import integral_image
from tqdm.auto import tqdm

def create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
                    max_feature_height):
    generated_features = []
    for feature in feature_types:
        feature_start_width = max(min_feature_width, feature_types[feature][0])
        for feature_width in range(feature_start_width, max_feature_width, feature_types[feature][0]):
            feature_start_height = max(min_feature_height, feature_types[feature][0])
            for feature_height in range(feature_start_height, max_feature_height, feature_types[feature][1]):
                for x in range(img_width - feature_width):
                    for y in range(img_height - feature_height):
                        generated_features.append(HaarLikeFeature(feature, y, x, feature_width, feature_height, 0, 1))
                        generated_features.append(HaarLikeFeature(feature, y, x, feature_width, feature_height, 0, -1))
    print("generated ", len(generated_features), " features")
    return generated_features


def learn(faces, non_faces, num_classifiers=-1, min_feature_height=1, max_feature_height=-1, min_feature_width=1,
          max_feature_width=-1):
    # list of training data with their labels and weights
    # for each feature find optimum threshold to minimise error
    # select best classifier
    # re-weight all data
    # repeat
    faces = list(map(integral_image, faces))
    non_faces = list(map(integral_image, non_faces))
    num_faces = len(faces)
    num_non_faces = len(non_faces)
    num_imgs = num_faces + num_non_faces
    img_height, img_width = faces[0].shape

    # default max feature height and width is same as that of image
    max_feature_height = img_height if max_feature_height == -1 else max_feature_height
    max_feature_width = img_width if max_feature_width == -1 else max_feature_width

    # creating data labels and weights
    data = faces + non_faces
    labels = np.hstack((np.ones(num_faces), np.ones(num_non_faces) * -1))
    weights = np.hstack((np.ones(num_faces) * (1 / (2 * num_faces)), np.ones(num_non_faces) * (1 / (2 * num_non_faces))))

    # create weak classifiers
    classifiers = create_features(img_height, img_width, min_feature_width, max_feature_width, min_feature_height,
                                  max_feature_height)

    # train
    best_classifiers = []
    for t in range(num_classifiers):  # required classifier
        print("searching for classifier", t+1, "of ", num_classifiers )
        # normalize the weights
        weights = weights / weights.sum()
        # for each classifier find error
        classification_errors = np.zeros((len(classifiers), num_imgs))
        for c in tqdm(range(len(classifiers))):  # generated classifiers
            classifier = classifiers[c]
            for i in range(num_imgs):
                h = classifier.predict(data[i])
                classification_errors[c][i] = weights[i] * abs(h - labels[i])
        # select best classifier
        total_errors = np.sum(classification_errors, axis=1)
        best_classifier_indx = np.argmin(total_errors)
        best_classifiers = np.append(best_classifiers, classifiers[best_classifier_indx])
        e = total_errors[best_classifier_indx]
        # set weight of classifier
        # update weights
        for i in range(num_imgs):
            if classification_errors[best_classifier_indx, i] == 0:
                weights[i] = weights[i] * e / (1 - e)
    return best_classifiers
