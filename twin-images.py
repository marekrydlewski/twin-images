# http://www.cs.put.poznan.pl/bwieloch/?page_id=653
from sklearn.metrics import roc_auc_score

def extract(image, keypoints):
    pass


def distance(descriptor1, descriptor2):
    pass


if __name__ == "__main__":
    y_true = [0, 0, 1, 1]  # 0 dla odpowiadających sobie punktów, 1 dla różnych
    y_scores = [0.1, 0.4, 0.35, 0.8]  # odległości zwrócone przez funkcję distance
    print(roc_auc_score(y_true, y_scores))

