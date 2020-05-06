#!/usr/bin/python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file',  help='path to input file (output of fairseq)')
parser.add_argument('--debpe',  action='store_true', help='enable the flag to post-process and remove BPE segmentation.')

args = parser.parse_args()


scount = -1
with open(args.input_file) as f:
    for line in f:
        line = line.strip()
        pieces = line.split('\t')
        scount = int(pieces[0][2:])
        if len(pieces) == 2: # output empty
            hyp = ''
        elif len(pieces) == 3:
            hyp = pieces[2]
        else:
            raise ValueError('invalid input')
        if args.debpe:
            hyp = hyp.replace('@@ ','')
        score = pieces[1]
        print("%d ||| %s ||| F0= %s ||| %s" % (scount, hyp, score, score) )


