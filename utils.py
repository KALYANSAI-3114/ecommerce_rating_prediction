from sklearn.preprocessing import LabelEncoder

class SafeLabelEncoder(LabelEncoder):
    def __init__(self):
        super().__init__()
        self.classes_set = set()

    def fit(self, y):
        super().fit(y)
        self.classes_set = set(self.classes_)
        self.classes_ = list(self.classes_) + ['unknown']
        return self

    def transform(self, y):
        y = [val if val in self.classes_set else 'unknown' for val in y]
        return super().transform(y)
