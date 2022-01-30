# Title: Deep learning classifier
# Author: Matteo Zullo
# Description: Flexibly and easily train most deep learning classifiers with GPU power

import pandas as pd
import numpy as np
from box import Box
import torch
import time
import logging
import datetime
import os
import sys
import shutil
import tempfile
import random

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy, fbeta

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, hamming_loss
from sklearn.model_selection import train_test_split
from ipywidgets import IntProgress



class classifier:
    
    
    def __init__(self, data, labels, model_type, model_name, cols: list):

        '''
        The function returns transformer-ready data.
        Numeric labels are decoded to improve interpretability.
        '''

        # import files
        labels = pd.read_csv(labels, encoding='latin-1')
        data = pd.read_csv(data, usecols = cols, encoding='latin-1')

        # standardize names
        data = data.rename(columns={cols[0]: 'index', cols[1]: 'text', cols[2]: 'label'}).set_index('index')

        # convert numeric labels to categories
        #data = data.reset_index().merge(labels, how="left", on = 'label').set_index('index')  # merge
        #data = data[['text','category']].rename(columns={"category": "label"})  # drop numeric labels
        
        self.data = data
        self.labels = labels
        self.model_type = model_type
        self.model_name = model_name
    
    def random_seed(seed = True) -> int:
        '''
        The function generates random seed.
        '''
        return random.randrange(10000)

    
    def split_data(self, data, seed, prop_test: float = 0.15, prop_validation: float = 0.15):
        '''
        The function returns 'fast-bert' ready data.
        fast-bert' multiclass requires data in the index-text-label format.
        Numeric labels are decoded to improve interpretability.
        '''
        
        # calculate split size
        n = len(data)
        n_test, n_validation = round(prop_test * n), round(prop_validation * n)
        n_train = n - (n_test +  n_validation)
        
        # split train-test-validation
        test = data.sample(n=n_test, replace=False, random_state=seed)
        validation = data.drop(test.index).sample(n=n_validation, replace=False, random_state=seed)
        train = data.drop(test.index).drop(validation.index)
        
        return train, test, validation

    
    def predict_model(self, learner, df_predictions):
        '''
        The function generates model predictions.
        '''
        
        # predict
        list_predictions = learner.predict_batch(df_predictions['text'].tolist())  # save list of predictions
        
        # size test, size of predictions
        n_obs, n_predictions = len(df_predictions), len(self.labels)
        
        # loop through predictions
        dict_predictions, dict_probabilities = {}, {}

        for i in range(n_predictions):
            # append predictions and probabilities to dicts
            dict_predictions["prediction_{}".format(i+1)] = [list_predictions[row][i][0] for row in range(n_obs)]
            dict_probabilities["probability_{}".format(i+1)] = [list_predictions[row][i][1] for row in range(n_obs)]

            # append predictions and probabilities to test data
            df_predictions["prediction_{}".format(i+1)] = dict_predictions["prediction_{}".format(i+1)]
            df_predictions["probability_{}".format(i+1)] = dict_probabilities["probability_{}".format(i+1)]

        return df_predictions   
    
    
    ###############################################################################
    ### INITIALIZE DATA ###
    ###############################################################################
    
    def main(self, 
             EP: int,  # number of epochs
             seed_model = True, seed_set = True,  # model and set seed
             prop_test = 0.15, prop_validation = 0.15,  # proportion data in test and validation
             optimal_lr = False  # optimize learning rate
            ):
    
        # initialize landing folders
        outputs = tempfile.TemporaryDirectory()
        inputs = tempfile.TemporaryDirectory()

        # initialize model seeds
        seed_model, seed_set = self.random_seed() if True else seed_model, self.random_seed() if True else seed_set

        # initialize model splits
        train, test, validation = self.split_data(self.data, seed_set, prop_test = 0.15, prop_validation = 0.15)
        dataframes, datafiles = [train, test, validation], ['train.csv', 'test.csv', 'valid.csv'] 

        for df, f in zip(dataframes, datafiles):
            df.to_csv(os.path.join(inputs.name, f), index=False)

        # save labels to inputs
        labels_unique = train.label.unique()
        labels_list, labelfile = list(labels_unique), 'label.csv'  # unique labels
        pd.DataFrame(labels_list).to_csv(os.path.join(inputs.name, labelfile), index=False, header=False)

        # initialize tensor for GPU
        use_gpu = torch.cuda.is_available()  # GPU available
        n_gpu = torch.cuda.device_count()  # GPU available (no.)
        FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
        Tensor = FloatTensor
        
        # define evaluation metrics
        def F1_macro(y_pred: Tensor, y_true: Tensor, average='macro', sample_weight=None):
            '''
            The function caculates F1 macro.
            '''  
            # if GPU is available
            if use_gpu:
                y_pred = np.argmax(y_pred.cpu(), axis=1)
                return f1_score(y_true.cpu(), y_pred.cpu(), average=average, sample_weight=sample_weight)

            # if GPU is not available
            else:
                y_pred = np.argmax(y_pred, axis=1)
                return f1_score(y_true, y_pred, average=average, sample_weight=sample_weight)


        def F1_micro(y_pred: Tensor, y_true: Tensor):
            '''
            The function caculates F1 Mico.
            '''

            # if GPU is available
            if use_gpu:
                return F1_macro(y_pred.cpu(), y_true.cpu(), average='micro')

            # if GPU is not available
            else:
                return F1_macro(y_pred, y_true, average='micro')


        def F1_by_class(y_pred: Tensor, y_true: Tensor, labels: list = labels_list, sample_weight=None):
            '''
            The function caculates F1 for each class.
            '''

            F1_by_class_d = {}

            # if GPU is available
            if use_gpu:
                y_pred = np.argmax(y_pred.cpu(), axis=1)
                for i in range(len(labels)):
                    F1_by_class_d[labels[i]] = f1_score(
                        y_true.cpu(), y_pred.cpu(), average='micro', labels=[i])

            # if GPU is not available
            else:
                y_pred = np.argmax(y_pred, axis=1)
                for i in range(len(labels)):
                    F1_by_class_d[labels[i]] = f1_score(
                        y_true, y_pred, average='micro', labels=[i])
            return F1_by_class_d

        # initialize evaluation metrics
        metrics_list = [accuracy, F1_macro, F1_micro]

        # model arguments
        args = Box({

            # random rseed
            "seed": seed_set,

            # CPU/GPU settings
            "device": torch.device('cuda' if use_gpu else 'cpu'),
            "multi_gpu": True if n_gpu > 1 else False,

            # input data
            "model_name": self.model_name,  # e.g., bert-base-cased, xlnet-large-cased, etc.
            "model_type": self.model_type,  # model type
            "data_train": datafiles[0],  # training file
            "data_test": datafiles[1],  # test file
            "data_valid": datafiles[2],  # validation file
            "data_label": labelfile,  # label file
            # "input_text": "text",  text column  # change if not 'text'
            # "input_label": "label",  label column  # change if not 'label'
            "max_seq_length": 256,  # max number of characters
            "multi_label": False,  # set to True if multi-label, False if multi-class
            # "do_lower_case": True,  # uncomment to uncase text
            "batch_size": BS,  # batch size

            # log file
            "run_text": "Neural network classifier",

            # model evaluation
            # measures-of-fit
            "metrics": [{'name': metric.__name__, 'function': metric} for metric in metrics_list],

            # hyper-parameters
            "learning_rate": LR,  # learning rate
            "no_epochs": EP,  # no. of epochs
            "finetuned_wgts_path": None,  # location for finetuned language model
            "warmup_steps": WS,  # no. of warmup steps
            "fp16": False,  # floating point precision
            "logging_steps": 0,  # set a number to enable tensor flow
            "validate": True,  # validate model after each epoch
            "schedule": W,  # learning rate progress
            "optimizer": "lamb",  # optimizer: 'lamb', 'adamw'
            "weight_decay": D
        })


        ###############################################################################
        ### DEFINE LOG FILE ###
        ###############################################################################

        # initialize log file
        FILE = 'model{}_set{}'.format(seed_model, seed_set)

        # set up log file
        logging.root.handlers = []  # reinitialize handlers
        logging.basicConfig(
            format ='[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s',
            datefmt ='%H:%M:%S',
            filemode ='w',  # refresh at each run
            level = logging.INFO,  # set threshold level
            filename = FILE + '.log'  # initialize log file
        )


        logger = logging.getLogger()
        logging.info(args.model_type + " model specification")

        # log model specifications
        logging.info(args)

        # log labels
        logging.info("\n")
        logging.info("Number of labels: {}".format(len(labels_list)))
        logging.info("Labels: {}".format(labels_list))

        # log GPU stats
        logging.info("\n")
        if use_gpu:
            logging.info("Available GPUs: = {}".format(n_gpu))
            for gpu in range(n_gpu):
                logging.info(torch.cuda.get_device_name(gpu))
        else:
            logging.info("No available GPUs.")


        ###############################################################################
        ### DEFINE MODEL ###
        ###############################################################################

        # databunch
        databunch = BertDataBunch(
            inputs.name,  # data directory
            inputs.name,  # label directory
            tokenizer=args.model_name,  # model name
            model_type=args.model_type,  # model type
            train_file=args.data_train,  # training set
            test_data=args.data_test,  # test set
            val_file=args.data_valid,  # validation set
            label_file=args.data_label,  # label data
            # text_col=args.input_text,  # text column
            # label_col=args.input_label,  # label column
            max_seq_length=args.max_seq_length,  # max number of characters
            multi_label=args.multi_label,  # multi-label
            multi_gpu=args.multi_gpu,  # multiple GPU
            batch_size_per_gpu=args.batch_size  # batch size
        )

        # learner
        learner = BertLearner.from_pretrained_model(
            databunch,  # databunch object
            pretrained_path=args.model_name,  # model name
            metrics=args.metrics,  # measures-of-fit
            device=args.device,  # CPU/GPU settings
            logger=logger,  # logger
            output_dir=outputs.name,  # output directory
            finetuned_wgts_path=args.finetuned_wgts_path,  # location for finetuned language model
            warmup_steps=args.warmup_steps,  # no. of warmup steps
            multi_gpu=args.multi_gpu,  # multiple GPU
            is_fp16=args.fp16,  # floating point precision
            multi_label=args.multi_label,  # multi-label
            logging_steps=args.logging_steps  # tensor flow
        )

        ###############################################################################
        ### FIND OPTIMAL LEARNING RATE ###
        ###############################################################################

        if optimal_lr == True:
            learner.lr_find(
                start_lr = 1e-9, end_lr=10,  # bounds
                use_val_loss = True,
                optimizer_type = args.optimizer,
                num_iter = 200,
                step_mode = 'exp',
                smooth_f = 0.05,
                diverge_th = 5
            )

            return learner.plot(skip_start=10, skip_end=5, log_lr=True, show_lr=1e-3, ax=None)


        ###############################################################################
        ### FIT MODEL ###
        ###############################################################################

        # train model
        logging.info('\n')
        logging.info("Starting training...")
        start_fit = time.time()
        fit = learner.fit(
            epochs =args.no_epochs,  # no. of epochs
            lr = args.learning_rate,  # learning rate
            validate = args.validate, 	# validate model after each epoch
            schedule_type = args.schedule,  # learning rate progress
            optimizer_type = args.optimizer  # optimizer type
        )

        end_fit = time.time()
        logging.info("Finishing training.")
        logging.info("Training time: {}".format(end_fit - start_fit))

        # evaluate model
        logging.info('\n')
        logging.info("Starting BERT testing...")
        start_test = time.time()
        learner.validate()
        end_test = time.time()
        logging.info("Finishing BERT testing.")
        logging.info("Testing time: {}".format(end_test - start_test))

        # save model
        learner.save_model()


        ###############################################################################
        ### PREDICT ###
        ###############################################################################

        predictions = self.predict_model(learner, self.data)
        predictions.to_csv(FILE + 'PRED' + '.csv', index=True)
        #results = results(predictions)
        #results.to_csv(FILE + 'RESULTS'+ '.csv')

        # delete landing folders
        for directory in inputs, outputs:
            directory.cleanup()

# train
BERT = classifier('data.csv', 'labels.csv', model_type = 'xlnet', model_name = 'xlnet-base-cased', cols = ['id','text','Positive'])
BERT.main(EP = 1, seed_model = True, seed_set = True, prop_test = 0.15, prop_validation = 0.15, optimal_lr = False)
