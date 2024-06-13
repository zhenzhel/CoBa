from alignscore import AlignScore
import pickle
import joblib
import os
from tqdm import tqdm
import argparse
import numpy as np

def main(args):

    # checkpoint paths
    if args.eval_large:
        print("Evaluate on Alignscore-large.")
        scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0', ckpt_path=args.ckpt_path, evaluation_mode='nli_sp')
    else:
        print("Evaluate on Alignscore-base.")
        scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path=args.ckpt_path, evaluation_mode='nli_sp')

    os.makedirs(args.output_dir, exist_ok=True)

    key = 'gen_text'

    with open(args.file_name, 'rb') as f:
        results = pickle.load(f)
        contexts = []
        claims = []

        # keep track of the indices that are not valid
        invalid_idx = []
        for i,result in enumerate(results): 
            if result[key].strip() == '':
                invalid_idx.append(i)
                continue
            contexts.append(result['document'])
            claims.append(result[key])
    f.close()

    score = scorer.score(contexts=contexts, claims=claims)

    # if the summary is empty then the score is zero
    for i in invalid_idx:
        score.insert(i,0)

    if args.eval_large:
        score_type = 'large'
    else:
        score_type = 'base'

    save_name = os.path.basename(args.file_name).replace(".pkl", "")
    joblib.dump(score, os.path.join(args.output_dir, f'{save_name}_alignscore_{score_type}_score.pkl'))
    print('file dumped.')

    print(f'alignscore-{score_type}: {args.file_name}: {np.mean(score)}')

def get_arg_parser():
    parser = argparse.ArgumentParser(description="evaluating")
    parser.add_argument("--ckpt_path", type=str, help="path to model checkpoint")
    parser.add_argument("--output_dir", type=str, help='path to save results')
    parser.add_argument("--file_name", type=str, help='path to the generated summaries')
    parser.add_argument("--eval_large", action='store_true', default=False, help='whether or not to evaluate on alignscore-large')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arg_parser()
    main(args)

