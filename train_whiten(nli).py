# coding: utf-8
"""
用AllNLI计算得到whitening的kernel和bias.

@env: python3, pytorch>=1.7.1, transformers==4.2.0
@author: Weijie Liu
@date: 20/01/2020
"""
import os
import random

from all_utils import (build_model, compute_kernel_bias, save_whiten,
                       sents_to_vecs)

NLI_PATH = './data/AllNLI.tsv'

MODEL_NAME_LIST = [
    './model/bge-base-zh',
    './model/bge-large-zh',
]

POOLINGS = [
    'cls',
    'first_last_avg'
]
# POOLING = 'last_avg'
# POOLING = 'last2avg'

MAX_LENGTH = 512
OUTPUT_DIR = './whiten/'


def load_dataset(path):
    """
    loading AllNLI dataset.
    """
    senta_batch, sentb_batch = [], []
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()[1:]
        for i, line in enumerate(random.sample(lines, 50_000)):
            items = line.strip().split('\t')
            senta, sentb = items[-3], items[-2]
            senta_batch.append(senta)
            sentb_batch.append(sentb)
    return senta_batch, sentb_batch


def main():

    a_sents_train, b_sents_train  = load_dataset(NLI_PATH)
    print("Loading {} training samples from {}".format(len(a_sents_train), NLI_PATH))

    for MODEL_NAME in MODEL_NAME_LIST:
        tokenizer, model = build_model(MODEL_NAME)
        print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))

        for pooling in POOLINGS:
            print("Transfer sentences to BERT vectors.")
            a_vecs_train = sents_to_vecs(a_sents_train, tokenizer, model, pooling, MAX_LENGTH)
            b_vecs_train = sents_to_vecs(b_sents_train, tokenizer, model, pooling, MAX_LENGTH)

            print("Compute kernel and bias.")
            kernel, bias = compute_kernel_bias([a_vecs_train, b_vecs_train])

            model_name = MODEL_NAME.split('/')[-1]
            output_filename = f"{model_name}-{pooling}-whiten(NLI).pkl"
            if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            save_whiten(output_path, kernel, bias)
            print("Save to {}".format(output_path))


if __name__ == "__main__":
    main()
