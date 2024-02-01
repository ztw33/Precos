from tqdm import tqdm
import faiss
import faiss.contrib.torch_utils

import torch
from torch.utils.data import TensorDataset

import multiprocessing
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, T5EncoderModel


import argparse
import time
import os
import logging
import numpy as np
from pathlib import Path

from config import add_args
from utils import *

logger = logging.getLogger(__name__)
logger.setLevel(20)

def encode(args, model, eval_data):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    if args.save_dstore:
        dstore_filename = f'{args.output_dir}/dstore_keys.npy'
        if os.path.exists(dstore_filename):
           os.remove(dstore_filename)
        
        Path(dstore_filename).parent.mkdir(parents=True, exist_ok=True)
        dstore_keys = np.memmap(dstore_filename, dtype=np.float16, mode='w+', shape=(len(eval_sampler), args.dimension))
    
    elif args.subset:
        # setup faiss
        index_name = f'{args.dstore_dir}/dstore_index.indexed'
        cpu_index = faiss.read_index(index_name, faiss.IO_FLAG_ONDISK_SAME_DIR)
        cpu_index.nprobe = 32
        if args.faiss_gpu:
            start = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index, co)
            logger.info(f'Moving index to GPU took {time.time() - start} s')
        else:
            gpu_index = cpu_index
        index = gpu_index

        subset_idx_filename = f'{args.output_dir}/subset_idx.txt'
        Path(subset_idx_filename).parent.mkdir(parents=True, exist_ok=True)

        # dstore_filename = f'{args.output_dir}/dstore_keys.npy'
        # dstore_keys = np.memmap(dstore_filename, dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))


    if args.save_dstore:
        dstore_idx = 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Encoding"):
        source_ids = batch[0].to(args.device)
        with torch.no_grad():
            outputs = model(input_ids=source_ids)
            last_hidden_states = outputs.last_hidden_state
            batch_context_vector = last_hidden_states.mean(dim=1)  # (batch_size, hidden_size)

            batch_time_size = batch_context_vector.shape[0]
            if args.save_dstore:
                dstore_keys[dstore_idx:(batch_time_size + dstore_idx)] = batch_context_vector.cpu().numpy().astype(np.float16)
                dstore_idx += batch_time_size
            
            elif args.subset:
                queries = batch_context_vector
                if not args.faiss_gpu:
                    queries = queries.cpu()

                # search the faiss index
                _, I = index.search(queries, args.topk)  # (batch_size, topk)
                # save the results
                with open(subset_idx_filename, 'a') as f:
                    np.savetxt(f, I.cpu().numpy(), fmt='%d')



def build_index(args, num_keys_to_add_at_a_time=100000):
    index_name = f'{args.output_dir}/dstore_index.indexed'
    quantizer = faiss.IndexFlatL2(args.dimension)
    index = quantizer

    dstore_filename = f'{args.output_dir}/dstore_keys.npy'
    dstore_keys = np.memmap(dstore_filename, dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))

    start = 0
    while start < args.dstore_size:
        end = min(args.dstore_size, start + num_keys_to_add_at_a_time)
        to_add = dstore_keys[start:end].copy()
        index.add(torch.tensor(to_add.astype(np.float32)))
        start += num_keys_to_add_at_a_time

        if (start % 100000) == 0:
            logger.info(f'Added {start} tokens so far')
            logger.info(f'Writing Index {start}')
            faiss.write_index(index, f'{index_name}')

    logger.info(f'Writing Index to {index_name}')
    faiss.write_index(index, f'{index_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    if os.path.exists(f'{args.output_dir}/subset_idx.txt'):
        os.remove(f'{args.output_dir}/subset_idx.txt')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, local_files_only=True)
    model = T5EncoderModel.from_pretrained(args.model_name_or_path, local_files_only=True)

    args.dimension = model.config.hidden_size

    if "CSN" in args.data_filename and args.data_filename.endswith(".jsonl"):
        examples = read_CSN_examples(args.data_filename, data_num=-1)
    else:
        examples = read_project_examples(args.data_filename)

    if args.build_index:
        args.dstore_size = len(examples)
        build_index(args)
        exit(0)




    tuple_examples = [(example, tokenizer, args) for example in examples]

    cpu_cont = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_cont)

    if args.key == "code":
        src_ids = pool.map(convert_examples_to_src_ids, tqdm(tuple_examples, total=len(tuple_examples)))
    elif args.key == "nl":
        src_ids = pool.map(convert_examples_to_tgt_ids, tqdm(tuple_examples, total=len(tuple_examples)))
    else:
        raise ValueError("Invalid key.")
    pool.close()
    pool.join()

    all_source_ids = torch.tensor([src_id for src_id in src_ids], dtype=torch.long)
    data = TensorDataset(all_source_ids)

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    model.to(args.device)

    encode(args, model, data)

    




