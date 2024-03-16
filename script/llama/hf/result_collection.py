import argparse

import numpy as np
import pandas as pd

from utils import get_module_logger, load_specified_files, makedirs_recursive


def arg_parser():
    """
    Add arguments to main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sp",
        "--save_path",
        help="Directory for saving extracted info",
    )
    parser.add_argument(
        "-rp",
        "--result_path",
        help="Directory for few shot result",
    )
    parser.add_argument(
        "-pp",
        "--prompt_path",
        help="Directory for source prompts",
    )
    args = parser.parse_args()
    return args


def extract_info(texts: list, prompts: list):
    out = []
    for text, prompt in zip(texts, prompts):
        out.append(text.replace(prompt, ""))
    return out


def main():
    args = arg_parser()
    logger = get_module_logger(__name__)

    # makedirs_recursive(args.save_path)
    results, result_nums = load_specified_files(args.result_path, file_type="result", extension="txt")
    prompts, _ = load_specified_files(args.prompt_path, file_type="prompt", extract_nums=result_nums, extension="txt")
    logger.info(f"Loaded {len(results)} results")

    assert len(results) == len(prompts)
    assert set(results.keys()) == set(prompts.keys())

    output = {
        "image_id": [],
        "prompt": [],
        "generated_text": [],
        "extracted_info": [],
        "few_shot_num": [],
    }

    for ky in results.keys():
        iids = np.repeat(ky, len(results[ky]))
        extracted = extract_info(results[ky], prompts[ky])
        output["image_id"].extend(iids)
        output["few_shot_num"].extend(result_nums[ky])
        output["prompt"].extend(prompts[ky])
        output["generated_text"].extend(results[ky])
        output["extracted_info"].extend(extracted)

    logger.info("Extracted generated info")

    df_output = pd.DataFrame(output)
    df_output.to_csv(f"{args.save_path}/extracted_info.csv", index=False)

    logger.info("Info saved")

    return output


if __name__ == "__main__":
    _ = main()
