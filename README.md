# Correction with Backtracking Reduces Hallucination in Summarization

This is the official repo for [Correction with Backtracking Reduces Hallucination in Summarization](https://arxiv.org/abs/2310.16176).

## Abstract

Abstractive summarization aims at generating natural language summaries of a source document that are succinct while preserving the important elements. Despite recent advances, neural text summarization models are known to be susceptible to hallucinating (or more correctly confabulating), that is to produce summaries with details that are not grounded in the source document. In this paper, we introduce a simple yet efficient technique, CoBa, to reduce hallucination in abstractive summarization. The approach is based on two steps: hallucination detection and mitigation. We show that the former can be achieved through measuring simple statistics about conditional word probabilities and distance to context words. Further, we demonstrate that straight-forward backtracking is surprisingly effective at mitigation. We thoroughly evaluate the proposed method with prior art on three benchmark datasets for text summarization. The results show that CoBa is effective and efficient in reducing hallucination, and offers great adaptability and flexibility.


## Environment

To set up the environment:

```
conda create --name coba python=3.8
conda activate coba
pip install -r requirements.txt
```

Note that the `transformers` version **4.30.2** is important. The codebase is based on this version, and may be incompatible with other versions.

## Running CoBa

The files are organized as follows:

`summarize.py`: Main script for generating summaries using CoBa or other decoding methods.

`coba_generator.py`: Implementation of CoBa decoding

### Example command

greedy+CoBa:
```
python summarize.py \
	--model_name google/flan-t5-xl \
	--ckpt_path [llama checkpoint path (only used when model name is llama)] \
	--dataset_name newsroom \
	--subset_size 5000 \
	--subset_seed 101 \
	--dataset_path [newsroom dataset path (only used when dataset is newsroom)] \
	--use_coba \
	--coba_prob_thresh 0.2 
```

greedy+CoBa-d: additionally include
```
--coba_include_dist --coba_dist_thresh 0.5
```

nucleus+[CoBa/CoBa-d]: additionally include
```
--do_sample --top_p 0.9
```

context-aware+[greedy/nucleus]+[CoBa/CoBa-d]: additionally include
```
--adjust_by_unconditional --adjust_by_unconditional_scale 0.5
```

***Caveats***: Currently CoBa has been tested for the settings included in the paper, i.e.:

For `flan-t5`: 
- greedy/nucleus + CoBa/CoBa-d
- context-aware versions of the above

For `llama`: 
- greedy + CoBa/CoBa-d
- context-aware version of the above

It is recommended to use batch size 1 for CoBa decoding (i.e. without the `--use_batch` flag; see below). We provide a batchified implementation for greedy+CoBa (`greedy_search_coba_batchified`) as a reference, but it has only been tested at limited scale and empirically does not provide speed up compared to the unbatchified version.

### Arguments
Model args:
- `--model_name`:  `t5-[version]`, `google/flan-t5-[version]` (following the huggingface naming format) or `llama`

- `--ckpt_path`: Only used if model is `llama`, which needs to be manually downloaded. The path should look like `/some/path/llama-2-7b-chat-hf-converted/`

Dataset args:
- `--dataset_name`: One of `newsroom`, `cnn_dailymail` or `xsum`
- `--subset_size`: The subset size (if evaluating only on a random subset of the data)
- `--subset_seed`: Optional seed for subset sampling, for reproducibility
- `--start_index`: Optionally, only evaluate on a slice of either the full dataset or the random subset starting at index `start_index`
- `--end_index`: Optionally, only evaluate on a slice of either the full dataset or the random subset ending at index `end_index` (exclusive)
- `--dataset_path`: Only used for `newsroom`, which needs to be downloaded manually. The path should look like `/some/path/newsroom-release`

Logging args:
- `--output_path`: Folder to save results and logs

Decoding args:
- `--use_coba`: Flag for using CoBa 
- `--coba_prob_thresh`: Threshold for token probability used for CoBa backtracking
- `--coba_include_dist`: Flag for using `coba-d`, i.e. also considering token embedding distance
- `--coba_dist_thresh`: Threshold for token embedding distance used for CoBa
    
- `--adjust_by_unconditional`: Flag for doing context-aware decoding, which modulates the context-conditioned token probability by the unconditional token probability
- `--adjust_by_unconditional_scale`: Scaling hyperparameter for context-aware decoding
- `--coba_max_steps`: Upper limit for the number of forward+backtracking steps CoBa can perform for a single generation
- `--max_seq_len`: Max context length
- `--gen_min_length`: Minimum generation length
- `--gen_max_length`: Maximum generation length
- `--seed`: Optional seed for nucleus decodings, for reproducibility
- `--do_sample`: Flag for performing nucleus decoding
- `--top_p`: Hyperparameter for nucleus decoding
- `--use_batch`: Use the batchified version for greedy+CoBa; not recommended
- `--batch_size`: If `--use_batch` is set, the batch size to use

## Evaluation

Please refer to `eval_scripts/README_evaluation.md`.