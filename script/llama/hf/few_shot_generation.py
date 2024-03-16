import argparse
import glob
import os
from pathlib import Path
import yaml

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

from utils import get_module_logger, makedirs_recursive


def arg_parser():
    """
    Add arguments to main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sp",
        "--save_path",
        help="Directory for saving result",
    )
    parser.add_argument(
        "-pp",
        "--prompts_path",
        help="Directory for prompts",
    )
    parser.add_argument(
        "-cp",
        "--config_path",
        help="Directory for config.yaml",
    )
    parser.add_argument(
        "-msl",
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum size for prompt sequence",
    )
    parser.add_argument(
        "-ss",
        "--shot_size",
        type=int,
        default=5,
        help="Shot size for learning",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-13b-chat-hf",
        help="Model name on HuggingFace format",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If true, only the first batch is passed",
    )
    parser.add_argument(
        "--print_result",
        action="store_true",
        help="If true, result text is printed",
    )
    parser.add_argument(
        "-v",
        "--verbose_num",
        type=int,
        default=10,
        help="verbose num for logging",
    )
    args = parser.parse_args()
    return args


def read_prompts(path: str, max_seq_len: int):
    out = []
    file = f"{path}/prefix.txt"
    if os.path.isfile(file):
        with open(file, "r") as f:
            prefix = f.read()
    else:
        prefix = ""
    files = glob.glob(f"{path}/prompt*.txt")
    file_nums = []
    for file in sorted(files):
        file_num = file.split("/prompt")[-1].split(".txt")[0]
        with open(file, "r") as f:
            prompt = prefix + f.read()
        if len(prompt) > max_seq_len:
            pass
        else:
            out.append(prompt)
            file_nums.append(file_num)
    return out, file_nums


def list_iid_dirs(path):
    return os.listdir(path)


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def load_pipeline(model_name, config):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config_pl = config["pipeline"]["build"]

    pipeline = transformers.pipeline(
        config_pl["task"],
        model=model_name,
        torch_dtype=torch.float16,
        device_map=config_pl["device_map"],
    )
    return tokenizer, pipeline


def run_pipeline(prompts, pipeline, config, tokenizer, args):
    config_rn = config["pipeline"]["run"]
    out = pipeline(
        prompts,
        eos_token_id=tokenizer.eos_token_id,
        max_length=args.max_seq_len,
        output_attentions=True,
        return_dict_in_generate=True,
        **config_rn,
    )
    return out


def save_results(
    results,
    image_id,
    save_path,
    file_nums,
    print_result=False,
):
    for result, file_num in zip(results, file_nums):
        if type(result) == list:
            result_out = result[0]
        else:
            result_out = result
        makedirs_recursive(f"{save_path}/{image_id}")
        if print_result:
            print("\n==================================\n")
            print(f"Result for image_id {image_id} prompt #{file_num}")
            print("\n----------------------------------\n")
            print(f"generated_text: \n{result_out['generated_text']}")
            print("\n==================================\n")
        with open(f"{save_path}/{image_id}/result{file_num}.txt", "a", encoding="utf-8") as f:
            f.write(result_out["generated_text"])
        torch.save(
            result_out["attentions"].cpu(),
            f"{save_path}/{image_id}/attention{file_num}.pt",
        )
    return


def main():
    args = arg_parser()
    logger = get_module_logger(__name__)

    makedirs_recursive(args.save_path)
    test_iids = list_iid_dirs(args.prompts_path)
    config = load_config(f"{args.config_path}/{args.model_name.split('/')[-1]}.yaml")
    tokenizer, pipeline = load_pipeline(args.model_name, config)

    outputs = []
    for i, iid in enumerate(test_iids):
        verbose = not i % args.verbose_num
        if verbose:
            logger.info(f"Experiment #{i} for image_id {iid} started")
        prompts, file_nums = read_prompts(
            f"{args.prompts_path}/{iid}",
            args.max_seq_len,
        )
        if verbose:
            logger.info(f"{len(prompts)} prompts loaded")
        output = run_pipeline(prompts, pipeline, config, tokenizer, args)
        outputs.append(output)
        if verbose:
            logger.info("Prompting finished")
        _ = save_results(
            output,
            iid,
            args.save_path,
            file_nums,
            print_result=args.print_result and verbose,
        )
        if verbose:
            logger.info("Result saved")
        if args.debug:
            exit()

    return output


if __name__ == "__main__":
    _ = main()
