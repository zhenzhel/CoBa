# Evaluation Instructions

The evaluation command and environment information for each metric are provided in the sections below. Unfortunately different metrics may need to have separate environments due to library version conflicts.

**Input Format**: The inputs to all evaluations should be a pickle file `.pkl`. It contains a list of dictionaries, where each dictionary corresponds to a specific document/summary pair containing the following keys and values:

- `id`: document id
- `document`: the document to be summarized in plain text
- `summary`: the groundtruth summary in plain text
- `gen_ids`: a torch tensor of shape `1xn` containing the token ids of a generated summary
- `gen_text`: generated summary in plain text

Example pickle files can be found in `example_pkls.zip` under GitHub *Releases*.

## ROUGE, BertScore and Bleurt Score

**Main Eval Script**: These three metrics can be evaluated using `eval-base.py`. 

**Environment**: The corresponding environment can be built from `eval_base_requirements.txt`.


**Example Command**: 

```
python eval-base.py --file_name [file_name] --output_dir [output_dir] --bleurt_path [bleurt_path]
```

We use the `BLEURT-20` checkpoint, which can also be found in the official repo. `--bleurt_path` should look like `/some/path/BLEURT-20`.

**Output Format**: Outputs will be saved to `[file_name]_base_metrics.pkl`. It is a dictionary with structure 

```
{
  src: {k1:v1, k2:v2, ...},
  ref: {k1:v1, k2:v2, ...}
}
```

`src`: metrics for comparing source document and generated summary

`ref`: metrics for comparing reference summary and generated summary

`k1`, `k2`, ...: each is a metric name, e.g. `ROUGE-L`, `bert_precision`, etc.

`v1`, `v2`, ...:each is a numpy array, where each entry corresponds to the metric value for a specific document (or reference summary)/generated summary pair. 

Aggregated metric can be computed with `v1.mean()`, etc., which are also printed at the end of the evaluation.

## AlignScore

Cloned from the [official AlignScore repo](https://github.com/yuh-zha/AlignScore).

**Main Eval Script**: `eval-alignscore.py` 

**Environment**: Please follow the instructions in the [official AlignScore repo](https://github.com/yuh-zha/AlignScore).



**Example Command**:

```
python eval-alignscore.py --output_dir [output_dir] --file_name [file-name] --ckpt_path [ckpt_path]
```

By default, we use the base version of the alignscore model (therefore we do not set the `--eval_large` flag). `--ckpt_path` is the path to the alignscore model, which can be downloaded from the official repo. It should look like `/some/path/AlignScore-base.ckpt`.

**Output Format**: Output is saved to `[file_name]_alignscore_[base/large]_score.pkl`. It is a 1-d list of numbers, where each number corresponds to the AlignScore of a specific document/summary pair.

## FactCC

Modified from the [official FactCC repo](https://github.com/salesforce/factCC/tree/master).

**Main Eval Script**: `factcc/write_jsonl.py` and `factcc/run.py`

**Environment**: PLease follow the instructions in the [official FactCC repo](https://github.com/salesforce/factCC/tree/master).

**Example Command**:Wwe need to first convert the result pickles to jsonl format using the command below:

```
python write_jsonl.py --file_name [file_name] --data_dir [data_dir]
```

- *`--file_name`*: path to the `.pkl` result file.

- *`--data_dir`*: folder to save the `.jsonl` file.

Then we run the evaluation:

```
python run.py \
  --task_name factcc_annotated \
  --do_eval \
  --eval_all_checkpoints \
  --do_lower_case \
  --overwrite_cache \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 64 \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --data_dir [data_dir] \
  --file_name [file_name]  \
  --output_dir [output_dir]
```

- *`--output_dir`*: Path for the model checkpoint files, and for the output `.txt` log file.
 
  By default, we use the **FactCC** model, not the FactCCX model. Model checkpoint files can be found in the official repo, and should be put directly under `output_dir`. For example, suppose `output_dir` is `factcc_outputs`. Then the model files would have paths `factcc_outputs/pytorch_model.bin`, `factcc_outputs/config.json`, etc.

- *`--file_name`*: Should end with `.jsonl`. It should be a single file name without the path to the folder that contains it.

- *`--data_dir`*: The path to the folder that contains the jsonl file.

  For example, if the jsonl file is saved to `/some/path/summaries.jsonl`, then it should be `--file_name summaries.jsonl --data_dir /some/path`.

**Output Format**: `[file_name]_factcc_eval_results.txt` logs the average `acc` metric.
