from main.unsupervised.tfidf_model import TfidfModel
from config import Config
import logging
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


logger = logging.getLogger("rc")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def train(args):
    tfidf_model = TfidfModel(args)
    with open(args.data_path, 'rb') as fin:
        dataset = pickle.load(fin)
    vectorizer_tfidf = tfidf_model.sklearn_build(dataset.train_set, ['q1_tokens', 'q2_tokens'])
    sim_score, truth = tfidf_model.compute_sim(vectorizer_tfidf, dataset.dev_set)
    preds = [1 if t > args.tfidf_threshold else 0 for t in sim_score]
    print(classification_report(truth, preds))
    

if __name__ == '__main__':
    train(Config)
    
