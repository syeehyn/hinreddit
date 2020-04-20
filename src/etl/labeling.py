import pandas as pd
from fast_bert.data_cls import BertDataBunch
from pathlib import Path
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
from fast_bert.prediction import BertClassificationPredictor

DATA_PATH = Path('/datasets/dsc180a-wi20-public/Malware/group_data/group_02/kaggle')
LABEL_PATH = Path('/datasets/dsc180a-wi20-public/Malware/group_data/group_02/kaggle')
OUTPUT_DIR = Path('../interim')

def create_databunch(datapath, labelpath, valfile = 'train.csv'):
    """
    tokenize the training and validation datasets according to chosen tokenizor and model, here we will be using BERT
    
    args:
        datapath - the path where training and validation datasets are saved
        labelpath - the path where label categories are saved
        valfile - the file name of validation file, default to the training set if not specified
    """
    return BertDataBunch(datapath, labelpath,
                        tokenizer='bert-base-uncased',
                        train_file='train.csv',
                        val_file=valfile,
                        label_file='labels.csv',
                        text_col='comment_text',
                        label_col=['toxic','severe_toxic','obscene','threat','insult','identity_hate'],
                        batch_size_per_gpu=16,
                        max_seq_length=512,
                        multi_gpu=True,
                        multi_label=True,
                        model_type='bert')

def create_learner(databunch, outputpath):
    """
    create learner with the tokenized data, in which the training and validation process are included
    
    args:
        databunch - tokenized data
        outputpath - where the model will stored after fitting
    """
    logger = logging.getLogger()
    device_cuda = torch.device("cuda")
    metrics = [{'name': 'accuracy', 'function': accuracy}]

    return BertLearner.from_pretrained_model(
                            databunch,
                            pretrained_path='bert-base-uncased',
                            metrics=metrics,
                            device=device_cuda,
                            logger=logger,
                            output_dir=outputpath,
                            finetuned_wgts_path=None,
                            warmup_steps=500,
                            multi_gpu=True,
                            is_fp16=True,
                            multi_label=False,
                            logging_steps=50)

def fit_save(learner):
    """
    fit the learner object then save the output model
    
    args:
        learner - the learner object fitted
    """
    learner.fit(epochs=6,
            lr=6e-5,
            validate=True, 
            schedule_type="warmup_cosine",
            optimizer_type="lamb")
    learner.save_model()
    
def predictor(modelpath, labelpath):
    """
    create predictor that use a model saved to predict other texts
    
    args:
        modelpath - the path where model is saved
        labelpath - the path where categories of labels are saved
    """

    predictor = BertClassificationPredictor(
                    model_path=modelpath,
                    label_path=labelpath, # location for labels.csv file
                    multi_label=False,
                    model_type='xlnet',
                    do_lower_case=False)

    return predictor

def label(predictor, texts):
    """
    label given text with the predictor
    
    args:
        predictor - the predictor used to classify text
        texts - a list of sentences that are being classified
    """
    return predictor.predict_batch(texts)

def label_csv(predictor, file, path):
    """
    label a whole firt layer post csv file
    
    args:
        predictor - the predictor used to classify text
        file - the name of subreddit csv file
        path - the output path where labeling results are saved
    """
    df = pd.read_csv(file)
    predictions = df.selftext.apply(lambda x: label(predictor, x)).to_frame()
    predictions.to_csv(path+file.split('.')[0]+'_labels.csv', index=False)