import evaluate
import pickle
import os
import argparse
from bleurt import score as bleurt_scorer
import tensorflow as tf
import numpy as np
from copy import deepcopy

'''
Script for evaluating ROUGE score, BertScore and Bleurt.

results_src: metrics for comparing the source document and the generated summary.
results_ref: metrics for comparing the groundtruth summary and the generated summary.

Saved outputs: Suppose file_name is [name].pkl, then the results are saved to [output_dir]/[name]_base_metrics.pkl.
'''

def get_arg_parser():
    parser = argparse.ArgumentParser(description="evaluating ROUGE, BertScore and Bleurt metrics")
    parser.add_argument("--file_name", type=str, required=True, help="path to the pickle file to eval")
    parser.add_argument("--output_dir", type=str, required=True, help="path to the save folder")
    parser.add_argument('--bleurt_path', type=str, required=True, help='path to the Bleurt checkpoint, should look like /some/path/BLEURT-20')
    return parser

def eval_three_metrics(filename, rouge, bertscore, bleurtscore):
    
    with open(filename, "rb") as f:
        results = pickle.load(f)
    
    sources = [res['document'] for res in results] # context
    hypotheses = [res['gen_text'] for res in results] # generated summary
    references = [res['summary'] for res in results] # gt summary

    results_template = {'rouge1': None, 'rouge2': None, 'rougeL': None, 'rougeLsum': None, 
                'bert_precision': None, 'bert_recall': None, 'bert_f1': None, 
                'bleurt': None}

    results_src = deepcopy(results_template)
    results_ref = deepcopy(results_template)

    ######## rouge ##########
    print("Evaluating ROUGE...", end="", flush=True)
    scores_src = rouge.compute(predictions=hypotheses, references=sources, use_stemmer=True, use_aggregator=False)
    scores_ref = rouge.compute(predictions=hypotheses, references=references, use_stemmer=True,use_aggregator=False)
    
    for k,v in scores_src.items():
        assert k in results_src
        results_src[k] = np.array(v)
    
    for k,v in scores_ref.items():
        assert k in results_ref
        results_ref[k] = np.array(v)
    print("done")

    ####### bert #########
    print("Evaluating BertScore...", end="", flush=True)
    scores_src = bertscore.compute(predictions=hypotheses, references=sources, lang='en')
    scores_ref = bertscore.compute(predictions=hypotheses, references=references,lang='en')
    
    for k,v in scores_src.items():
        if k == 'hashcode':
            continue
        assert "bert_" + k in results_src
        results_src["bert_" + k] = np.array(v)
    
    for k,v in scores_ref.items():
        if k == 'hashcode':
            continue
        assert "bert_" + k in results_ref
        results_ref["bert_" + k] = np.array(v)
    print("done")

    ######## bleurt #########
    print("Evaluating Bleurt...", end="", flush=True)
    scores_src = bleurtscore.score(references=sources, candidates=hypotheses, batch_size=50)
    scores_ref = bleurtscore.score(references=references, candidates=hypotheses, batch_size=50)

    results_src['bleurt'] = np.array(scores_src)
    results_ref['bleurt'] = np.array(scores_ref)
    print("done")

    return results_src, results_ref
    
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    physical_devices = tf.config.list_physical_devices('GPU')
    for i in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[i], True)

    rouge = evaluate.load('rouge')
    bertscore = evaluate.load("bertscore")
    bleurtscore = bleurt_scorer.BleurtScorer(args.bleurt_path)

    results_src, results_ref = eval_three_metrics(args.file_name, rouge, bertscore, bleurtscore)

    save_name = os.path.basename(args.file_name).replace(".pkl", "")
    save_path = os.path.join(args.output_dir, save_name + "_base_metrics.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"src": results_src, "ref": results_ref}, f)
    
    ####### print results #######
    print(f"====== Results for {args.file_name} ======")
    print("Comparing source document and generated summary:")
    for k,v in results_src.items():
        print(f"{k}:{v.mean():.3f}")
    print("Comparing ground truth summary and generated summary:")
    for k,v in results_ref.items():
        print(f"{k}:{v.mean():.3f}")
    print(f"Evaluation results saved to {save_path}.")

    return

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
