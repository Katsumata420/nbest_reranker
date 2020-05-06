#!/usr/bin/python

import time
import numpy as np
import codecs
import argparse
import math
import sys

# Initializing the logging module
import logging
import log_utils as L

# For feature functions

# For KenLM features
sys.path.insert(0, 'lib/kenlm_python/')
import kenlm


# For edit operations feature
from lib import levenshtein

# For Bert Score
from transformers import (AlbertConfig, AlbertForSequenceClassification,
AlbertTokenizer)
from transformers.data.processors.utils import InputExample, InputFeatures
import torch

logger = logging.getLogger(__name__)


ln10 = math.log(10)
class LM:
    def __init__(self, name, path, normalize=False, debpe=False):
        self.path = path
        c = kenlm.Config()
        c.load_method = kenlm.LoadMethod.LAZY
        self.model = kenlm.Model(path, c)
        self.name = name
        self.normalize = normalize
        self.debpe = debpe
        logger.info('Intialized ' + str(self.model.order) + "-gram language model: " + path)

    def get_name(self):
        return self.name

    def get_score(self, source, candidate, item_idx):
        if self.debpe:
            candidate = candidate.replace('@@ ','')
        lm_score = self.model.score(candidate)
        log_scaled = round(lm_score*ln10,4)
        if self.normalize == True:
            if len(candidate):
                return (log_scaled * 1.0 ) / len(candidate.split())
        return str(round(lm_score*ln10,4)) 

class SAMPLE:
    def __init__(self, name):
        self.name = name

    def get_score(self, source, candidate, item_idx):
        return str(0.5) 

class BertScore:
    def __init__(self, name, model_path, model_type):
        # model_path: dir (c.f. otput_dir in transformers)
        self.name = name
        self.model_type = model_type.lower()
        MODEL_CLASSES = {
            'albert' : (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
        }

        # task_name = 'ged-reg'
        
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(
            model_path,
            do_lower_case=True # Note!!
        )
        self.model = model_class.from_pretrained(
            model_path
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def _create_example(self, sentence):
        guid = 'test-{}'.format(sentence)
        text_a = sentence.strip()
        example = InputExample(guid=guid, text_a=text_a)

        return example

    def convert_examples_to_features(
        self,
        example,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
    ):


        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        feature = \
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
                )

        return feature

    def make_input(self, sentence):
        # Load data features from cache or dataset file
        example = self._create_example(sentence)
        feature = self.convert_examples_to_features(
            example,
            self.tokenizer,
            max_length=128,
            pad_on_left=bool(self.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
            pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
        )
        features = [feature]


        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        dataset = (all_input_ids, all_attention_mask, all_token_type_ids)
        return dataset


    def get_score(self, source, candidate, item_idx):
        input_data = self.make_input(candidate.lower())
        with torch.no_grad():
            inputs = {'input_ids': input_data[0].to(self.device), \
                    'attention_mask': input_data[1].to(self.device), \
                    'token_type_ids': input_data[2].to(self.device)}
            outputs = self.model(**inputs)
            logits = float(outputs[0].item())
        
        if logits > 1.0: # 1 means error sentence
            score = 1.0
        elif logits < 0.0:
            score = 0.0
        else:
            score = logits

        return str(score) 

class WordPenalty:
    '''
        Feature to caclulate word penalty, i.e. number of words in the hypothesis x -1
    '''
    def __init__(self, name):
        self.name = name

    def get_score(self, source, candidate, item_idx):
        return str(-1 * len(candidate.split())) 

class EditOps:
    '''
        Feature to calculate edit operations, i.e. number of deletions, insertions and substitutions
    '''
    def __init__(self, name, dels=True, ins=True, subs=True):
        self.name = name
        self.dels = ins
        self.ins = ins
        self.subs = subs

    def get_score(self, source, candidate, item_idx):
        src_tokens = source.split()
        trg_tokens = candidate.split()
        # Get levenshtein matrix
        lmatrix, bpointers = levenshtein.levenshtein_matrix(src_tokens, trg_tokens, 1, 1, 1)

        r_idx = len(lmatrix)-1
        c_idx = len(lmatrix[0])-1
        ld = lmatrix[r_idx][c_idx]
        d = 0 
        i = 0 
        s = 0 
        bpointers_sorted = dict()

        for k, v in bpointers.items():
            bpointers_sorted[k] =sorted(v, key=lambda x: x[1][0])

        # Traverse the backpointer graph to get the edit ops counts
        while (r_idx != 0 or c_idx != 0): 
            edit = bpointers_sorted[(r_idx,c_idx)][0]
            if edit[1][0] == 'sub':
                s = s+1
            elif edit[1][0] == 'ins':
                i = i+1
            elif edit[1][0] == 'del':
                d = d+1
            r_idx = edit[0][0]
            c_idx = edit[0][1]
        scores = ""
        if self.dels:
            scores += str(d) + " "
        if self.ins:
            scores += str(i) + " "
        if self.subs:
            scores += str(s) + " "
        return scores

class LexWeights:
    '''
    Use translation model from SMT p(w_f|w_e) using the alignment model from NMT
    '''
    def __init__(self, name, f2e=None, e2f=None, align_file=None, debpe=False):

        self.name = name
        if align_file:
            logger.info("Reading alignment file")
            self.align_dict = self.prepare_align_dict(align_file, debpe) 
        self.f2e_dict = None     
        if f2e:
            logger.info("Reading lex f2e file: " + f2e)
            self.f2e_dict =  self.prepare_lex_dict(f2e)
        self.e2f_dict = None
        if e2f:
            logger.info("Reading lex e2f file: " + e2f)
            self.e2f_dict = self.prepare_lex_dict(e2f)

        #for k in sorted(self.align_dict.iterkeys()):
            #print k, ":", self.align_dict[k].shape

    def set_align_file(align_file, debpe):
        logger.info("Reading alignment file")
        self.align_dict = self.prepare_align_dict(align_file, debpe)

    def prepare_lex_dict(self, lex_file):
        lex_dict = dict()
        with open(lex_file) as f:
            for line in f:
                pieces = line.strip().split()
                lex_dict[(pieces[0],pieces[1])] = math.log(float(pieces[2]))
        return lex_dict    

    def prepare_align_dict(self, align_file, debpe):
        sent_count = -1
        item_count = 0
        align_dict = dict()
        aligns = []
        src_sent = ""
        candidate_sent = ""
        count = 1
        with open(align_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pieces = line.split('|||')
                if len(pieces) > 1:

                    aligns = np.array(aligns)

                    ## Utility function to debpe aligns
                    def debpe_aligns(aligns, src_sent, candidate_sent):
                        src_tokens = src_sent.split()
                        candidate_tokens = candidate_sent.split()

                        # debug
                        src_debpe_tokens = src_sent.replace('@@ ','').split()
                        cand_debpe_tokens = candidate_sent.replace('@@ ','').split()

                        #print src_tokens, candidate_tokens, aligns.shape
                        assert aligns.shape == (len(candidate_tokens)+1, len(src_tokens)+1) or aligns.shape == (len(candidate_tokens), len(src_tokens)+1) , "Mismatch before debpe!" + str(aligns.shape) + " " + src_sent + " ( " + str(len(candidate_tokens)) + " ) " + " CAND:" + candidate_sent
                        before_shape = aligns.shape
                        ### Summing up and averaging across rows (candidate tokens) where BPE split occurs
                        start_idx = -1
                        end_idx = -1
                        delete_rows = []
                        for i in xrange(len(candidate_tokens)):
                            cand_token = candidate_tokens[i]
                            if len(cand_token)>=2 and cand_token[-2:] == '@@':
                                if start_idx == -1:
                                    start_idx = i
                                    end_idx = i
                                else:
                                    end_idx = i
                            else:
                                if start_idx != -1:
                                    aligns[start_idx] = np.sum(aligns[start_idx:end_idx+2], axis=0) / (end_idx - start_idx + 2)
                                    delete_rows += range(start_idx+1, end_idx+2)
                                    start_idx = -1 

                        ### Summing up across columns (src_tokens) where BPE split occurs
                        start_idx = -1
                        end_idx = -1
                        delete_cols = []
                        for j in xrange(len(src_tokens)):
                            src_token = src_tokens[j]
                            if len(src_token) >= 2 and src_token[-2:]== '@@':
                                if start_idx == -1:
                                    start_idx = j
                                    end_idx = j
                                else:
                                    end_idx = j
                            else:
                                if start_idx != -1:
                                    aligns[:,start_idx] = np.sum(aligns[:, start_idx:end_idx+2], axis=1)
                                    delete_cols += range(start_idx+1, end_idx+2)
                                    start_idx = -1

                        #print aligns.shape, delete_rows, delete_cols
                        aligns = np.delete(aligns, delete_rows, axis=0)
                        aligns = np.delete(aligns, delete_cols, axis=1)

                        #print len(src_debpe_tokens), len(cand_debpe_tokens), aligns.shape, before_shape, src_tokens, src_debpe_tokens
                        #print src_tokens, len(src_tokens)
                        #print src_debpe_tokens, len(src_debpe_tokens)
                        #print candidate_tokens, len(candidate_tokens)
                        #print cand_debpe_tokens, len(cand_debpe_tokens)
                        #print before_shape, (len(candidate_tokens), len(src_tokens)), aligns.shape, (len(cand_debpe_tokens), len(src_debpe_tokens)) 
                        assert aligns.shape == (len(cand_debpe_tokens)+1, len(src_debpe_tokens)+1) or aligns.shape == (len(cand_debpe_tokens), len(src_debpe_tokens)+1), "mismatch after debpe!" + str(len(src_debpe_tokens))
                        return aligns

                    ### End of utility function ##

                    before_shape = aligns.shape
                    if sent_count>-1 and debpe == True:
                        aligns = debpe_aligns(aligns, src_sent, candidate_sent)
                    '''
                    if sent_count == 167 and item_count == 7:
                        #print aligns.shape
                        print "DEBUG"
                        src_debpe_tokens = src_sent.replace('@@ ','').split()
                        cand_debpe_tokens = candidate_sent.replace('@@ ','').split()
                        candidate_tokens = candidate_sent.split()
                        src_tokens = src_sent.split()
                        print src_debpe_tokens, len(src_debpe_tokens)
                        print candidate_tokens, len(candidate_tokens)
                        print cand_debpe_tokens, len(cand_debpe_tokens)
                        print before_shape, (len(candidate_tokens), len(src_tokens)), aligns.shape, (len(cand_debpe_tokens), len(src_debpe_tokens)) 
                    '''
                    align_dict[(sent_count, item_count)] = aligns
                    aligns = []
                    if int(pieces[0]) == sent_count:
                        item_count += 1
                    else:
                        assert sent_count + 1 == int(pieces[0]), "Malformed alignment file!"
                        sent_count =  sent_count+1
                        item_count = 0
                    src_sent = pieces[3]
                    candidate_sent = pieces[1]
                else:
                    weights = [float(piece) for piece in line.split()]
                    aligns.append(weights)

        aligns = np.array(aligns)
        if sent_count>-1 and debpe == True:
            aligns = debpe_aligns(aligns, src_sent, candidate_sent)
        align_dict[(sent_count, item_count)] = np.array(aligns)
        return align_dict
 
    def get_score(self, source, candidate, item_idx ):
        aligns = self.align_dict[item_idx]
        if (len(candidate.split())+1, len(source.split())+1) != aligns.shape and (len(candidate.split()), len(source.split())+1) != aligns.shape:
            print (source, candidate, aligns.shape, len(source.split()), len(candidate.split()))
        assert (len(candidate.split())+1, len(source.split())+1) == aligns.shape or (len(candidate.split()), len(source.split())+1) == aligns.shape, "Alignment dimension mismatch at: " + str(item_idx)
        candidate_tokens = candidate.split()
        source_tokens = source.split()
        f2e_score = 0.0     
        e2f_score = 0.0
        for i in xrange(len(candidate_tokens)):
            for j in xrange(len(source_tokens)):
                #print "CANDIDATE_TOKEN:", candidate_tokens[i], "SOURCE_TOKEN:", source_tokens[j], "PROB:", self.f2e_dict[(candidate_tokens[i], source_tokens[j])], "ALIGN:", aligns[i,j]
                if self.f2e_dict:
                    if (candidate_tokens[i], source_tokens[j]) in self.f2e_dict:
                        f2e_score += self.f2e_dict[(candidate_tokens[i], source_tokens[j])]*aligns[i,j]
                    else:
                        f2e_score += math.log(0.0000001)*aligns[i,j]
                if self.e2f_dict:
                    if (source_tokens[j], candidate_tokens[i]) in self.e2f_dict:
                        e2f_score += self.e2f_dict[(source_tokens[j], candidate_tokens[i])]*aligns[i,j]
                    else:
                        e2f_score += math.log(0.0000001)*aligns[i,j]
        scores = ""
        if self.f2e_dict:
            scores += str(f2e_score) + " "
        if self.e2f_dict:
            scores += str(e2f_score) + " "

        return scores
