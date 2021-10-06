import logging
import os
import re
import sys

import time
from argparse import ArgumentParser


import warnings

import tqdm

from chunk_generator_context import chunk

warnings.simplefilter(action='ignore', category=FutureWarning)


from simpletransformers.t5 import T5Model, T5Args

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


if __name__ == '__main__':
    parser = ArgumentParser(description='')
    parser.add_argument('data_path', help='')
    parser.add_argument('save_path', help='')
    parser.add_argument('--model_type', default="mt5", help='Name the model')
    parser.add_argument('--model_name', default="google/mt5-base", help='Name the model')

    parser.add_argument('--batch_size', default=32, type=int, help='mini batch size')
    parser.add_argument('--max_seq_length', default=256, type=int, help='max_seq_length')
    parser.add_argument('--max_length', default=256, type=int, help='')
    parser.add_argument('--num_beams', default=1, type=int, help='')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help='')

    parser.add_argument('--chunk', default=20, type=int, help='chunk size')
    parser.add_argument('--stride', default=5, type=int, help='stride size')
    
    parser.add_argument('--top_k', default=50, type=int, help='top_k')
    parser.add_argument('--top_p', default=0.95, type=float, help='top_p')
    parser.add_argument('--no_sample', action='store_true', help='sample')
    
    cache_dir = os.getenv('SCRATCH', '.') + '/cache/'
    args = parser.parse_args()


    # Configure the model
    model_args = T5Args()


    model_args.fp16 = False  # dont work with True
    model_args.max_length = args.max_length  # 20
    model_args.num_beams = args.num_beams  # 1
    model_args.repetition_penalty = args.repetition_penalty  # 1.0

    model_args.train_batch_size = args.batch_size
    model_args.eval_batch_size = args.batch_size
    model_args.use_multiprocessed_decoding = False

    model_args.max_seq_length = args.max_seq_length

    # gramformer
    model_args.do_sample = not args.no_sample
    model_args.top_k = args.top_k
    model_args.top_p = args.top_p

    time_of_run = time.time()

    model = T5Model(args.model_type, args.model_name, args=model_args)
    # print(model.args)

    # TODO: predict

    

    # Make predictions with the model
    # to_predict = [
    #     'ocr: Uradowany, Neron złożył dzięki niebiosom, uniesiony wdzięcznością za opiekę bogów.',
    #     'ocr: Przybywszy do Rzymu nie zwołał ani ludu ani senatu.',
    # ]
    # 
    # preds = model.predict(to_predict)
    # 
    # print(preds)
    
    try:
        f = open(args.save_path)
        lines=f.readlines()
        lines=lines[:-2]
        f.close()
    except UnicodeDecodeError:
        print('Unicode error - delete last line')
        sys.exit()
    except:
        lines=[]
    
    print('Lines:', len(lines))
    
    f=open(args.save_path, 'w')
    f.writelines(lines)
    f.flush()
    
    for i, hyp_line in enumerate(tqdm.tqdm(open(args.data_path))):
        if i<len(lines): continue
        
        hyp_line = hyp_line[:-1].replace('-\\n', '').replace('\\n', '\n\\n').replace('\\\\', '\\')
        # doc_id, page_no, year, hyp = hyp_line.split('\t')
        hyp = hyp_line.split('\t')[-1]

        hyp = re.sub(r'(\\n| )+', ' ', hyp)
        hyp = hyp.split()

        xs=[]
        for context_left, start, end, context_right in chunk(hyp, args.chunk, args.stride):
            if args.stride > 0:
                x = ' '.join(hyp[context_left:start]).strip() + ' <left_context> ' + ' '.join(
                    hyp[start:end]).strip() + ' <right_context> ' + ' '.join(
                    hyp[end:context_right]).strip()
            else:
                x = ' '.join(hyp[start:end]).strip()

            x = re.sub(' +', ' ', x)
            xs.append(x)

        # print([f'ocr: {x}' for x in xs])
        preds = model.predict([f'ocr: {x}' for x in xs])

        f.write(' '.join(preds))
        f.write('\n')
        f.flush()