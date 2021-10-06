from argparse import ArgumentParser

import Levenshtein


# r = "private Thread currentThread; private  1 2 3 Thread currentThread; private Thread currentThread; private Thread currentThread; ".split()
# h = "private 3 Thread currentThread; private Thread currentThread; private Thread currentThread; private Thread currentThread; 1 2 3".split()


def levenshtein_editops_list(source, target):
    unique_elements = sorted(set(source + target))
    char_list = [chr(i) for i in range(len(unique_elements))]
    if len(unique_elements) > len(char_list):
        raise Exception("too many elements")
    else:
        unique_element_map = {ele: char_list[i] for i, ele in enumerate(unique_elements)}
    source_str = ''.join([unique_element_map[ele] for ele in source])
    target_str = ''.join([unique_element_map[ele] for ele in target])
    transform_list = Levenshtein.opcodes(source_str, target_str)
    return transform_list


def align(r, h):
    x_ref = []
    x_hyp = []
    x_act = []
    for x in levenshtein_editops_list(r, h):
        # print(x)
        tag, i1, i2, j1, j2 = x
        # print(tag, r[i1:i2], h[j1:j2])
        maxs = max(j2 - j1, i2 - i1)
        x_act.extend([tag] * maxs)
        x_ref.extend(r[i1:i2])
        x_ref.extend([''] * (maxs - (i2 - i1)))
        x_hyp.extend(h[j1:j2])
        x_hyp.extend([''] * (maxs - (j2 - j1)))
    return x_ref, x_hyp, x_act


def better_match(x_ref, x_hyp, x_act):
    # dla każdego tokenu sprawdź czy można go przesunąć na puste miejsce zmniejszając dystans

    # przesuwamy w ref
    for i in range(len(x_ref)):
        act = x_act[i]
        if act == 'equal': continue

        ref = x_ref[i]
        hyp = x_hyp[i]
        best_dist = Levenshtein.distance(ref, hyp)
        best_index = i

        j = i - 1
        while j >= 0:

            if x_ref[j] == '':
                new_dist = Levenshtein.distance(ref, x_hyp[j])
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_index = j
            #TODO błąd
            else:
                break
            j -= 1

        if best_index != i:
            x_ref[best_index] = ref
            x_ref[i] = ''
            if abs(best_index-i)>1:
                print(best_index, i)
                # print(x_hyp, x_ref)
                for i, (x, y) in enumerate(zip(x_hyp, x_ref)):
                    print(i,[x,y])

# for x in levenshtein_editops_list(r, h):
#     print(x)
#     tag, i1, i2, j1, j2=x
#     print(tag, r[i1:i2], h[j1:j2])

import glob
import re
import sys

import jsonlines
import tqdm

if __name__ == "__main__":
    parser = ArgumentParser(description='Align texts')
    parser.add_argument('in_path', help='path to in TSV')
    parser.add_argument('ref_path', help='path to expected TSV')
    parser.add_argument('align_path', help='path to output align_path JSONL')
    parser.add_argument('solution_path', help='path to output baseline solution TSV')
    args = parser.parse_args()

    with open(args.solution_path, 'w') as f, jsonlines.open(args.align_path, mode='w') as writer:
        for hyp_line, ref_line in tqdm.tqdm(zip(open(args.in_path), open(args.ref_path))):
            hyp_line = hyp_line[:-1].replace('-\\n', '').replace('\\n', '\n\\n').replace('\\\\', '\\')

            doc_id, page_no, year, hyp = hyp_line.split('\t')

            ref_line = ref_line[:-1].replace('\\n', '\n\\n').replace('\\\\', '\\')
            ref = ref_line

            hyp = re.sub(r'(\\n| )+', ' ', hyp)
            ref = re.sub(r'(\\n| )+', ' ', ref)
            # hyp=hyp.replace('-\\n','')

            ref = ref.split()
            hyp = hyp.split()
            # print(ref, hyp)

            x_ref, x_hyp, x_act = align(ref, hyp)
            better_match(x_ref, x_hyp, x_act)
            better_match(x_hyp, x_ref, x_act)

            # x = levenshtein_editops_list(ref, hyp)

            writer.write((x_ref, x_hyp))

            f.write(' '.join(hyp))
            f.write('\n')

            # break
