import integralimage

feature_types = {
    1: (1, 2),  # "Two vertical"
    2: (2, 1),  # "Two horizontal"
    3: (1, 3),  # "three vertical"
    4: (3, 1),  # "three horizontal"
    5: (2, 2),  # "four quadrants"
}


class HaarLikeFeature:
    """
    HaarLikeFeature object
    """

    def __init__(self, feature_type, y, x, width, height, threshold, polarity):
        """

        :param feature_type: 1,2,3,4,5
        :param y: top left corner y
        :param x: top left corner x
        :param self.width: width of entire feature/filter
        :param self.height: height of entire feature/filter
        :param threshold: feature threshold
        :param polarity: -1 or 1
        """
        self.feature = feature_types[feature_type]
        self.y = y
        self.x = x
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        self.weight = 1

    def get_feature_value(self, ii):
        """
        evaluates the feature value by subtracting the white region of the filter from its black region
        :param ii: integral image
        :return: feature value
        """
        feature_value = 0
        """
        b 
        w
        """
        if self.feature == 1:
            black = integralimage.sum_rect_feature(ii, self.y, self.x, self.height / 2, self.width)
            white = integralimage.sum_rect_feature(ii, self.y + self.height / 2, self.x, self.height / 2, self.width)
            feature_value = black - white
        """
        b w
        """
        if self.feature == 2:
            black = integralimage.sum_rect_feature(ii, self.y, self.x, self.height, self.width / 2)
            white = integralimage.sum_rect_feature(ii, self.y, self.x + self.width / 2, self.height, self.width / 2)
            feature_value = black - white
        """
        b1 w b2
        """
        if self.feature == 3:
            black1 = integralimage.sum_rect_feature(ii, self.y, self.x, self.height, self.width / 3)
            white = integralimage.sum_rect_feature(ii, self.y, self.x + self.width / 3, self.height, self.width / 3)
            black2 = integralimage.sum_rect_feature(ii, self.y, self.x + 2 * self.width / 3, self.height,
                                                    self.width / 3)
            feature_value = black1 - white + black2
        """
        b1
        w
        b2
        """
        if self.feature == 4:
            black1 = integralimage.sum_rect_feature(ii, self.y, self.x, self.height / 3, self.width)
            white = integralimage.sum_rect_feature(ii, self.y + self.height / 3, self.x, self.height / 3, self.width)
            black2 = integralimage.sum_rect_feature(ii, self.y + 2 * self.height / 3, self.x, self.height / 3,
                                                    self.width)
            feature_value = black1 - white + black2
        """
        b1 w1
        w2 b2
        """
        if self.feature == 5:
            black1 = integralimage.sum_rect_feature(ii, self.y, self.x, self.height / 2, self.width / 2)
            white1 = integralimage.sum_rect_feature(ii, self.y, self.x + self.width / 2, self.height / 2,
                                                    self.width / 2)
            white2 = integralimage.sum_rect_feature(ii, self.y + self.height / 2, self.x, self.height / 2,
                                                    self.width / 2)
            black2 = integralimage.sum_rect_feature(ii, self.y + self.height / 2, self.x + self.width / 2,
                                                    self.height / 2, self.width / 2)
            feature_value = black1 - white1 - white2 + black2

        return feature_value

    def predict(self, ii):
        h = self.get_feature_value(ii)
        return self.weight * (1 if self.polarity * h < self.polarity * self.threshold else 0)
