# coding: utf-8
"""
使用target训练得到的whitening参数, 在下游任务测试.

@env: python3, pytorch>=1.7.1, transformers==4.2.0
@author: Weijie Liu
@date: 20/01/2020
"""
import logging

import numpy as np
from prettytable import PrettyTable

import senteval
import senteval.engine
from all_utils import *

MAX_LENGTH = 512
BATCH_SIZE = 1024
TEST_PATH = './data/'
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


MODEL_LIST = [
    "./model/bge-base-zh",
    "./model/bge-large-zh",
]

POOLINGS = [
    "cls",
    "first_last_avg",
]

MODEL_ZOOS = {}

for model in MODEL_LIST:
    model_name = model.rsplit("/", 1)[-1]
    model_name_show = ''.join([s.capitalize() for s in model_name.replace("-zh", "").split("-")])
    model_scale = model_name.split("-")[1]

    d_original = 768 if model_scale == "base" else 1024
    d_reduced = 256 if model_scale == "base" else 384

    for pooling in POOLINGS:
        for d in [d_original, d_reduced]:
            MODEL_ZOOS[f"{model_name_show}-whiten-{d}(NLI)-{pooling}"] = {
                "encoder": model,
                "pooling": pooling,
                "whiten_file": f"./whiten/{model_name}-{pooling}-whiten(NLI).pkl",
                "n_components": d
            }


def prepare(params, samples):
    return None


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = []
    for sent in batch:
        vec = sent_to_vec(sent, params['tokenizer'], \
                params['encoder'], params['pooling'], MAX_LENGTH)
        embeddings.append(vec)
    embeddings = np.vstack(embeddings)
    embeddings = transform_and_normalize(embeddings,
            kernel=params['whiten'][0],
            bias=params['whiten'][1]
        )  # whitening
    return embeddings


def run(model_name, test_path):

    model_config = MODEL_ZOOS[model_name]
    logging.info(f"{model_name} configs: {model_config}")

    tokenizer, encoder = build_model(model_config['encoder'])
    logging.info("Building {} tokenizer and model successfuly.".format(model_config['encoder']))

    kernel, bias = load_whiten(model_config['whiten_file'])
    kernel = kernel[:, :model_config['n_components']]
    logging.info("Loading kernel and bias from {}".format(model_config['whiten_file']))

    # Set params for senteval
    params_senteval = {
            'task_path': test_path,
            'usepytorch': True,
            'tokenizer': tokenizer,
            'encoder': encoder,
            'pooling': model_config['pooling'],
            'whiten': (kernel, bias),
            'batch_size': BATCH_SIZE
        }

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = [
            'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
            'STSBenchmarkCosin',
            'SICKRelatednessCosin',
        ]
    results = se.eval(transfer_tasks)

    # Show results
    table = PrettyTable(["Task", "Spearman"])
    for task in transfer_tasks:
        if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            metric = results[task]['all']['spearman']['wmean']
        elif task in ['SICKRelatednessCosin', 'STSBenchmarkCosin']:
            metric = results[task]['spearman']
        table.add_row([task, metric])
    logging.info(f"{model_name} results:\n" + str(table))


def run_all_model():

    for model_name in MODEL_ZOOS:
        run(model_name, TEST_PATH)


if __name__ == "__main__":
    # run('BERTbase-whiten-256(NLI)', TEST_PATH)
    run_all_model()
