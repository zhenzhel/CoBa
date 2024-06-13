import os

from datasets import load_dataset

from tqdm import tqdm
from coba_generator import CoBaGenerator
import argparse
import pickle
import torch
import numpy as np
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import random
from datetime import datetime

def get_arg_parser():
    parser = argparse.ArgumentParser(description="summarization")

    parser.add_argument("--model_name", required=True, help="currently supporting t5-[version], google/flan-t5-[version] or llama")
    parser.add_argument('--ckpt_path', default="llama models need to be manually downloaded. the path should look like /some/path/llama-2-7b-chat-hf-converted/")

    parser.add_argument("--dataset_name", required=True, choices=['newsroom', 'xsum', 'cnn_dailymail'])
    parser.add_argument("--subset_size", default=None, type=int)
    parser.add_argument("--subset_seed", default=None, type=int)
    parser.add_argument("--dataset_path", default="", help='newsroom dataset needs to be downloaded manually. the path should look like /some/path/newsroom-release')

    
    parser.add_argument("--output_path", default='decoding_results')

    parser.add_argument("--use_coba", action='store_true', default=False, help='if set, use coba decoding')
    parser.add_argument('--coba_prob_thresh', type=float, default=0.2)
    parser.add_argument('--coba_include_dist', action='store_true', default=False, help='if set, backtrack by both uncertainty and similarity. Otherwise only by uncertainty.')
    parser.add_argument('--coba_dist_thresh', type=float, default=0.5)
    
    parser.add_argument('--adjust_by_unconditional', action='store_true', default=False)
    parser.add_argument('--adjust_by_unconditional_scale', type=float, default=0.5)

    parser.add_argument('--coba_max_steps', type=int, default=2000)
    

    parser.add_argument('--max_seq_len', type=int, default=1024, help="context length")
    parser.add_argument('--use_batch', action='store_true', default=False)
    parser.add_argument("--batch_size", default=1, type=int)

    parser.add_argument('--gen_min_length', type=int, default=2, help="min generation length")
    parser.add_argument('--gen_max_length', type=int, default=200, help="max generation length")
    parser.add_argument('--seed', type=int, default=None, help='optional seed for nucleus decoding')
    parser.add_argument('--do_sample', action='store_true', default=False, help='for nucleus decoding')
    parser.add_argument('--top_p', default=0.9, type=float, help='for nucleus decoding')

    return parser

def set_all_random_seeds(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def log_config(desc, confs, log_name, mode):
    f = open(log_name, mode)
    if desc is not None:
        print(desc)
        f.write(desc + "\n")
    
    for k,v in vars(confs).items():
        print(f"{k}: {v}")
        f.write(f"{k}: {v}\n")
    f.close()
    return

def main(args):
    set_all_random_seeds(args.seed)

    MAX_SEQ_LEN = args.max_seq_len

    filename = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    os.makedirs(args.output_path, exist_ok=True)

    log_name = f"{args.output_path}/{filename}.txt"
    log_config("Experiment config:", args, log_name, "w")

    save_name = f"{args.output_path}/{filename}.pkl"

    if args.dataset_name == 'newsroom':
        dataset = load_dataset(args.dataset_name, data_dir=args.dataset_path)
        KEY_DOCUMENT, KEY_GT_SUMMARY, KEY_ID = 'text', 'summary', 'url'
    elif args.dataset_name == "cnn_dailymail":
        dataset = load_dataset(args.dataset_name, '3.0.0')
        KEY_DOCUMENT, KEY_GT_SUMMARY, KEY_ID = 'article', 'highlights', 'id'
    elif args.dataset_name ==  "xsum":
        dataset = load_dataset("xsum")
        KEY_DOCUMENT, KEY_GT_SUMMARY, KEY_ID = 'document', 'summary', 'id'
    else:
        raise NotImplementedError(f'Dataset not supported: {args.dataset_name}')

    if args.model_name.startswith("t5-") or args.model_name.startswith("google/flan-t5-"):
        model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_name, device_map='auto')
        tokenizer = transformers.T5Tokenizer.from_pretrained(args.model_name, model_max_length=MAX_SEQ_LEN)
    elif args.model_name.startswith('llama'):
        model = LlamaForCausalLM.from_pretrained(args.ckpt_path, device_map='auto')
        tokenizer = LlamaTokenizer.from_pretrained(args.ckpt_path, model_max_length=MAX_SEQ_LEN)
        tokenizer.pad_token = tokenizer.decode([32000-1])
    else:
        raise NotImplementedError(f'Model type not supported: {args.model_name}')
    
    model.config.n_positions = MAX_SEQ_LEN
    tokenize_fn = lambda x: tokenizer(x, max_length=MAX_SEQ_LEN, return_tensors='pt', padding=True, truncation=True)

    gen = CoBaGenerator(model=model, tokenizer=tokenizer, 
                             use_coba=args.use_coba, 
                             coba_prob_thresh=args.coba_prob_thresh,
                             coba_include_dist = args.coba_include_dist, 
                             coba_dist_thresh = args.coba_dist_thresh, 
                             adjust_by_unconditional = args.adjust_by_unconditional, 
                             adjust_by_unconditional_scale=args.adjust_by_unconditional_scale, 
                             coba_max_steps = args.coba_max_steps,
                             use_batch=args.use_batch,
                             llama=args.model_name == 'llama'
                             )
    gen._dist_reset(args.batch_size)
        
    if args.model_name.startswith('llama'):
        generation_config = transformers.GenerationConfig.from_pretrained(args.ckpt_path)
        new_generation_confs = {
            'max_new_tokens': 200, 
            'min_length': 0, 
            'num_beams': 1, 
            'do_sample': False,
            'pad_token_id': tokenizer.pad_token_id,
        }
        assert not args.do_sample, "sampling not yet supported for llama"
    elif args.model_name.startswith("t5-") or args.model_name.startswith("google/flan-t5-"):
        generation_config = transformers.GenerationConfig.from_pretrained(args.model_name)
        new_generation_confs = {
            'max_length': args.gen_max_length,
            'min_length': args.gen_min_length, 
            'num_beams': 1, 
            'do_sample': args.do_sample
        }
        if args.do_sample:
            new_generation_confs['top_p'] = args.top_p
    else:
        raise NotImplementedError

    generation_config.update(**new_generation_confs)
    
    log_config("\nGeneration config:", generation_config, log_name, "a")
    
    generation_results = []
    if args.dataset_name == 'newsroom':
        ds_length = len(dataset['test'][KEY_ID])
    else:
        ds_length = len(dataset['test'])
    indices = np.arange(ds_length)
    
    if args.subset_size is not None:
        rng = np.random.default_rng(seed=args.subset_seed)
        indices = rng.choice(indices, size=args.subset_size, replace=False)
        indices = np.sort(indices)

    for cur_index in tqdm(range(0, len(indices), args.batch_size)):
        doc_indices = indices[cur_index:cur_index+args.batch_size]
        if (args.end_index != -1 and cur_index >= args.end_index) or cur_index < args.start_index:
            continue

        if args.dataset_name == 'newsroom':
            CONTEXT = [dataset['test'][KEY_DOCUMENT][doc_idx] for doc_idx in doc_indices]
        elif args.dataset_name == 'xsum':
            CONTEXT = [dataset['test'][int(doc_idx)][KEY_GT_SUMMARY] + " " + dataset['test'][int(doc_idx)][KEY_DOCUMENT] for doc_idx in doc_indices]
        elif args.dataset_name == 'cnn_dailymail':
            CONTEXT = [dataset['test'][int(doc_idx)][KEY_DOCUMENT] for doc_idx in doc_indices]
        else:
            raise NotImplementedError
        
        # add prompt
        if args.model_name.startswith("t5-") or args.model_name.startswith("google/flan-t5-"):
            CONTEXT = ["summarize: " + doc for doc in CONTEXT]
        
        CONTEXT_OBJ = tokenize_fn(CONTEXT)
        if args.model_name == 'llama':
            assert args.batch_size == 1
            if not args.use_coba:
                CONTEXT_OBJ['input_ids'] = torch.concat((CONTEXT_OBJ['input_ids'][:, :MAX_SEQ_LEN-11], torch.tensor([[13, 13, 11139,  3034,  675, 297, 472, 1556, 1023, 25260, 29901]])), dim=1)
                CONTEXT_OBJ['attention_mask'] = torch.ones(1, CONTEXT_OBJ['input_ids'].shape[-1])
            else:
                CONTEXT_OBJ['input_ids'] = torch.concat((CONTEXT_OBJ['input_ids'][:, :MAX_SEQ_LEN-11], torch.tensor([[13, 13, 11139,  3034,  675, 297, 472, 1556, 1023, 25260]])), dim=1)
                CONTEXT_OBJ['attention_mask'] = torch.ones(1, CONTEXT_OBJ['input_ids'].shape[-1])
        
        CONTEXT_OBJ = {k:v.cuda() for k,v in CONTEXT_OBJ.items()}

        if args.adjust_by_unconditional:
            if args.do_sample or args.batch_size > 1 or args.coba_prob_thresh !=0:
                assert not args.model_name.startswith('llama'), "llama with CAD has not been tested for backtracking, nucleus sampling or batch size >1"

            if args.model_name.startswith("t5-") or args.model_name.startswith("google/flan-t5-"):
                UNCOND_CONTEXT = ["summarize: " for _ in CONTEXT]
            elif args.model_name.startswith("llama"):
                UNCOND_CONTEXT = ["\n\nSummarize in at most two sentences:" for _ in CONTEXT]
            else:
                raise NotImplementedError

            UNCOND_CONTEXT_OBJ = tokenize_fn(UNCOND_CONTEXT)
            UNCOND_CONTEXT_OBJ = {k:v.cuda() for k,v in UNCOND_CONTEXT_OBJ.items()}
            gen.set_uncond_inputs(UNCOND_CONTEXT_OBJ['input_ids'], 
                                  attention_mask_uncond=UNCOND_CONTEXT_OBJ.get("attention_mask", None))

        if args.coba_include_dist:
            gen.precompute_context_embeddings(CONTEXT_OBJ['input_ids'])

        summary_gen = gen.generate(CONTEXT_OBJ['input_ids'],  
                                   attention_mask=CONTEXT_OBJ['attention_mask'],
                                   generation_config=generation_config,
                                    output_scores=False, return_dict_in_generate=False, output_attentions=False, 
                                    use_cache=False)
        texts = tokenizer.batch_decode(summary_gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        summary_gen = summary_gen.detach().cpu()

        iter_res = []
        for i,doc_idx in enumerate(doc_indices):
            
            doc_idx = int(doc_idx)
            if args.dataset_name == "newsroom": 
                cur_res = {
                    'id': dataset['test'][KEY_ID][doc_idx], 
                    'document': dataset['test'][KEY_DOCUMENT][doc_idx],
                    'summary': dataset['test'][KEY_GT_SUMMARY][doc_idx], 
                }
            elif args.dataset_name == 'xsum':
                cur_res = {
                    'id': dataset['test'][doc_idx][KEY_ID], 
                    'document': dataset['test'][doc_idx][KEY_GT_SUMMARY] + " " + dataset['test'][doc_idx][KEY_DOCUMENT], 
                    'summary': dataset['test'][doc_idx][KEY_GT_SUMMARY],
                }
            elif args.dataset_name == 'cnn_dailymail':
                cur_res = {
                    'id': dataset['test'][doc_idx][KEY_ID], 
                    'document': dataset['test'][doc_idx][KEY_DOCUMENT], 
                    'summary': dataset['test'][doc_idx][KEY_GT_SUMMARY],
                }
            else:
                raise NotImplementedError
            
            cur_res['gen_ids'] = summary_gen[i]
            cur_res['gen_text'] = texts[i]
            if args.model_name == 'llama':
                # sanitized output
                gen_text = texts[i]
                gen_text = gen_text.replace(':\n', '')
                if gen_text[0] == ':':
                    gen_text = gen_text[1:]
                gen_text = gen_text.strip()
                cur_res['gen_text'] = gen_text

            if gen._coba_steps is not None:
                cur_res['coba_steps'] = gen._coba_steps[i]
            
            iter_res.append(cur_res)
        generation_results.extend(iter_res)

        gen._dist_reset(args.batch_size)
        gen.reset_context_embedding_matrix()
    
        with open(save_name, "wb") as f:
            pickle.dump(generation_results, f)

    print(f"Results saved to {save_name}")
    return

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
