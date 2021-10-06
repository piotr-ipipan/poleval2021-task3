import re
from argparse import ArgumentParser

import jsonlines
import numpy as np
import tqdm
from transformers import AutoTokenizer


def chunk(A, size, stride):
    for i in range(0, len(A), size):
        start = i
        end = min(len(A), i + size)
        context_left = max(0, start - stride)
        context_right = min(len(A), end + stride)

        yield context_left, start, end, context_right


if __name__ == "__main__":
    parser = ArgumentParser(description='Align texts')
    parser.add_argument('align_path', help='path to align_path JSONL')
    parser.add_argument('train_path', help='path to train TSV')
    parser.add_argument('--chunk', default=20, type=int, help='chunk size')
    parser.add_argument('--stride', default=5, type=int, help='stride size')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")

    x_lengths = []
    y_lengths = []

    with jsonlines.open(args.align_path) as reader, open(args.train_path, 'w') as f:
        for x_ref, x_hyp in tqdm.tqdm(reader):
            # print(x_ref)
            # print(x_hyp)
            for context_left, start, end, context_right in chunk(x_hyp, args.chunk, args.stride):
                
                if args.stride>0:
                    x = ' '.join(x_hyp[context_left:start]).strip() + ' <left_context> ' + ' '.join(
                        x_hyp[start:end]).strip() + ' <right_context> ' + ' '.join(
                        x_hyp[end:context_right]).strip()
                else:
                    x = ' '.join(x_hyp[start:end]).strip()
                y = ' '.join(x_ref[start:end]).strip()
                x = re.sub(' +', ' ', x)
                y = re.sub(' +', ' ', y)
                
                if not x and not y:
                    # print(start, end, len(x_ref))
                    continue
                    
                # print(f"{x}\t{y}")
                # print(tokenizer(x))
                x_lengths.append(len(tokenizer(x)['input_ids']))
                y_lengths.append(len(tokenizer(y)['input_ids']))
                f.write(f"{x}\t{y}\n")

    print('Input and output size in subtokens:')
    print(f'Input mean: {np.mean(x_lengths)}, max: {np.max(x_lengths)}, min: {np.min(x_lengths)}')
    print(f'Output mean: {np.mean(y_lengths)}, max: {np.max(y_lengths)}, min: {np.min(y_lengths)}')

