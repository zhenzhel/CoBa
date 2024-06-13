import json 
import pickle 
import os
import argparse

def parse_story_file(content):
    """
    Remove article highlights and unnecessary white characters.
    """
    content_raw = content.split("@highlight")[0]
    content = " ".join(filter(None, [x.strip() for x in content_raw.split("\n")]))
    return content

def get_arg_parser():
    parser = argparse.ArgumentParser(description="evaluating")
    parser.add_argument("--file_name", type=str, help='path to the generated summaries')
    parser.add_argument("--data_dir", type=str, help='path to save results')
    return parser.parse_args()

def main(args):
    os.makedirs(args.data_dir,exist_ok=True)

    with open(args.file_name,'rb') as f:
        results = pickle.load(f)

    jsonl_save_name = os.path.basename(args.file_name).replace(".pkl", ".jsonl")
    output_file = os.path.join(args.data_dir, jsonl_save_name)
    with open(output_file, "w") as fd:
        for result in results:
            example = dict()
            example['claim'] = result['gen_text']
            example['text'] = parse_story_file(result['document'])
            example['label'] = 'CORRECT'
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")   
   
    print(f"Results saved to {args.data_dir}.")
    return

if __name__ == "__main__":
    args = get_arg_parser()
    main(args)



