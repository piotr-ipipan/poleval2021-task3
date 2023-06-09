import logging
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1" #do debuggingu
import random
import sys
import time
from argparse import ArgumentParser
from typing import List

import warnings

import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

import Levenshtein
import numpy as np
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

torch.set_num_threads(2)


def wer_list(source:List[str],target:List[str]):
    # average wrod WER between lists of strings
    results=[]
    for r,h in zip(source, target):
        res=wer(r.split(), h.split())
        # print(f'{res}: {r} --- {h}', file=sys.stderr)
        results.append(res)
    return float(np.mean(results))

def wer(source: List[str],target:List[str]):
    # word WER on tokenized string
    unique_elements = sorted(set(source + target))
    char_list = [chr(i) for i in range(len(unique_elements))]
    if len(unique_elements) > len(char_list):
        raise Exception("too many elements")
    else:
        unique_element_map = {ele: char_list[i] for i, ele in enumerate(unique_elements)}
    source_str = ''.join([unique_element_map[ele] for ele in source])
    target_str = ''.join([unique_element_map[ele] for ele in target])
    transform_list = Levenshtein.distance(source_str, target_str)

    return 0.0 if max(len(source), len(target))==0 else transform_list / max(len(source), len(target))

def load_data(path) -> (str, str):
    for line in open(path):
        x,y=line[:-1].split('\t')
        yield x, y

if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('train_path', help='')
    parser.add_argument('eval_path', help='')
    parser.add_argument('--model_type', default="mt5", help='Name the model')
    parser.add_argument('--model_name', default="google/mt5-base", help='Name the model')
    #parser.add_argument('--model_name', default="google/mt5-small", help='Name the model') ############ MOJE
    
    parser.add_argument('--datasets',
                        help='delimited datasets input: all_80_20, all_batch124_80_20, batch124_80_20, all_50_50, all_batch124_50_50, batch124_50_50',
                        type=str, default='all_80_20')
    parser.add_argument('--wandb_project', default='sh-v5', help='Project name in wandb')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--epochs', default=5, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='mini batch size')
    parser.add_argument('--early_stopping_metric', default='wer', help='early_stopping_metric')
    parser.add_argument('--acc', default=1, type=int, help='gradient_accumulation_steps')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    # parser.add_argument('--eps', default=1e-8, type=float, help='adam eps')
    parser.add_argument('--gradient', default=1.0, type=float, help='gradient')
    # parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay (L2 penalty)')
    parser.add_argument('--max_seq_length', default=256, type=int, help='max_seq_length')
    # parser.add_argument('--weights', default=1.0, type=float, help='weights of 1 to balance')
    parser.add_argument('--evaluate_during_training_steps', default=100, type=int, help='evaluation period')
    parser.add_argument('--warmup_steps', default=100, type=int, help='weights of 1 to balance')
    parser.add_argument('--max_length', default=256, type=int, help='')
    parser.add_argument('--num_beams', default=1, type=int, help='')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help='')
    parser.add_argument('--scheduler', default='constant_schedule_with_warmup', help='')

    #cache_dir = os.getenv('SCRATCH', '.') + '/cache/' #############MOJE
    cache_dir = '/home/pborkowski/storage/polevaltask3_cache/' #############MOJE
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    train_data=[['ocr', x, y] for x,y in load_data(args.train_path)]
    # train_data = [
    #     # ["generate question", "Star Wars is an American epic space-opera media franchise created by George Lucas, which began with the eponymous 1977 film and quickly became a worldwide pop-culture phenomenon", "Who created the Star Wars franchise?"],
    #     # ["generate question", "Anakin was Luke's father" , "Who was Luke's father?"],
    #   ['ocr','wróżba dodala mu po drodze otuchy.', 'wróżba dodała mu po drodze otuchy'],
    #   ['ocr','Oczy jego padły na płaskorzeżbę na pomniku,', 'Oczy jego padły na płaskorzeźbę na pomniku,'],
    #   ['ocr','wyobrażającą Galla zwyciężo: nego przez rzymskiego rycerza.', 'wyobrażającą Galla zwyciężonego przez rzymskiego rycerza.'],
    #   ['ocr','Uradowany, Neron złożył dzięki niebiosom, uniesiony wdzięcznością za opiekę bogów.', 'Uradowany Neron złożył dzięki niebiosom, uniesiony wdzięcznością za opiekę bogów.'],
    # ]
    train_df = pd.DataFrame(train_data)
    train_df.columns = ["prefix", "input_text", "target_text"]

    eval_data = [['ocr', x, y] for x, y in load_data(args.eval_path)]
    # eval_data = [
    #     # ["generate question", "In 2020, the Star Wars franchise's total value was estimated at US$70 billion, and it is currently the fifth-highest-grossing media franchise of all time.", "What is the total value of the Star Wars franchise?"],
    #     # ["generate question", "Leia was Luke's sister" , "Who was Luke's sister?"],
    #   ['ocr','wróżba dodala mu po drodze otuchy.', 'wróżba dodała mu po drodze otuchy'],
    #   ['ocr','Oczy jego padły na płaskorzeżbę na pomniku,', 'Oczy jego padły na płaskorzeźbę na pomniku,'],
    #   ['ocr','wyobrażającą Galla zwyciężo: nego przez rzymskiego rycerza.', 'wyobrażającą Galla zwyciężonego przez rzymskiego rycerza.'],
    #   ['ocr','Uradowany, Neron złożył dzięki niebiosom, uniesiony wdzięcznością za opiekę bogów.', 'Uradowany Neron złożył dzięki niebiosom, uniesiony wdzięcznością za opiekę bogów.'],
    #   ['ocr','Przybywszy do Rzymu nie zwołał ani ludu ani senatu.','Przybywszy do Rzymu nie zwołał ani ludu ani senatu.'],
    # ]
    eval_df = pd.DataFrame(eval_data)
    eval_df.columns = ["prefix", "input_text", "target_text"]
    
    # Configure the model
    model_args = T5Args()

    ###caly ponizszy blok dodalem sam ###
    model_args.use_multiprocessing = False ############### MOJE 
    #model_args.num_workers = 3 ############### MOJE
    model_args.use_multiprocessing_for_evaluation = False  # #######ODKOM JAK ZA MALO PAMIECI
    # model_args.set_num_threads = 26     ############### MOJE
    # model_args.overwrite_output_dir = True ############### MOJE
    model_args.process_count = 2
    model_args.cache_dir = cache_dir
    ###############

    model_args.num_train_epochs = args.epochs
    # model_args.no_save = True
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True
    model_args.overwrite_output_dir = True
    model_args.save_eval_checkpoints=False
    model_args.save_model_every_epoch=False
    model_args.fp16 = False # dont work with True
    model_args.max_length=args.max_length #20
    model_args.num_beams=args.num_beams #1
    model_args.repetition_penalty=args.repetition_penalty #1.0
    model_args.scheduler=args.scheduler

    model_args.wandb_project = args.wandb_project
    model_args.manual_seed = args.seed
    model_args.train_batch_size = args.batch_size
    model_args.eval_batch_size = args.batch_size
    model_args.early_stopping_metric = args.early_stopping_metric
    model_args.use_multiprocessed_decoding = False
    
    model_args.gradient_accumulation_steps = args.acc
    model_args.learning_rate = args.learning_rate
    # model_args.adam_epsilon = args.eps
    # model_args.max_grad_norm = args.gradient
    # model_args.weight_decay = args.weight_decay
    model_args.max_seq_length = args.max_seq_length

    # model_args.early_stopping_metric_minimize = False
    # model_args.warmup_steps = args.warmup_steps
    model_args.evaluate_during_training_steps = args.evaluate_during_training_steps
    
    #gramformer
    model_args.do_sample=True
    model_args.top_k = 50
    model_args.top_p = 0.95

    time_of_run = time.time()
    output_dir = cache_dir+'model_bin_' + args.model_name.replace('/', '_') + '_' + str( 
        time_of_run) ##MOJA MODYFIKACJA dodanie cache_dir+
    best_model_dir = output_dir + "/best_model/"

    model_args.output_dir = output_dir
    model_args.best_model_dir = best_model_dir
    
    model = T5Model(args.model_type, args.model_name, args=model_args)
    print(model.args)

    #TODO: predict
    
    # Train the model
    model.train_model(train_df, eval_data=eval_df, wer=wer_list)
    
    # Evaluate the model
    result = model.eval_model(eval_df, wer=wer_list)
    
    print(result)
    
    # Make predictions with the model
    to_predict = [
        'ocr: Uradowany, Neron złożył dzięki niebiosom, uniesiony wdzięcznością za opiekę bogów.',
        'ocr: Przybywszy do Rzymu nie zwołał ani ludu ani senatu.',
    ]
    
    preds = model.predict(to_predict)
    
    print(preds)