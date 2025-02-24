import random
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm

from evaluate import evaluate
from utils import set_seed
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from litellm import batch_completion
from jinja2 import Template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="debug", type=str)
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--api_key", default="sk-srxmnwhmfdmvuunnvksstijtcydrofpjdokdgfavvjwgaxuj", type=str)
    parser.add_argument("--model", default="openai/Qwen/Qwen2.5-14B-Instruct", type=str)
    parser.add_argument("--base_url", default="https://api.siliconflow.cn/v1", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--method", default="cot", type=str)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local model configuration",
    )
    parser.add_argument(
        "--parallel-execution",
        action="store_true",
        help="Use parallel execution for transform and solve steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing problems",
    )
    args = parser.parse_args()

    return args


def prepare_data(data_name, args):
    examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    output_dir = os.path.join(
        "./work_dirs", 
        datetime.now().strftime("%Y-%m-%d"), 
        args.exp, 
        data_name,
        args.method,
        datetime.now().strftime("%H-%M-%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    return examples, output_dir


def setup(args):
    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def solve_problems(args, samples, output_dir):
    if args.method == "cot":
        def construct_messages(question):
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."},
                {"role": "user", "content": question},
            ]
            return messages
        # repeat n times
        prompts = [
            construct_messages(sample["question"]) for sample in samples for _ in range(args.n_sampling)
        ]
        # get all outputs
        batch_completions = batch_completion(
            model=args.model,
            messages=prompts,
            temperature=args.temperature,
            stream=False,
            api_key=args.api_key,
            api_base=args.base_url,
        )
        outputs = [
            completion.choices[0].message.content 
            for completion in batch_completions
        ]
    elif args.method == "template":
        from src.tree import Tree 

        local_config = {
            "api_key": "EMPTY",
            "base_url": "http://qwen2.5-coder-32b-mindie.test.polaris:8080/v1",
            "model": "openai/qwen2.5_coder_32b",
            "temperature": 0.7,
        }

        llm_config = (
            local_config
            if args.local
            else {
                "api_key": args.api_key,
                "base_url": args.base_url,
                "model": args.model,
                "temperature": 0.7,
            }
        )

        template = Tree(
            work_dirs=output_dir,
            template="/root/Projects/meta_r1/MetaCoT/src/templates/msc2020.json",
            **llm_config,
            parallel_execution=args.parallel_execution,
        )

        # Extract all questions and their IDs for batch processing
        questions_with_ids = [
            (sample["idx"], sample["question"]) for sample in samples
        ]

        outputs = []
        for batch in [
            questions_with_ids[i : i + args.batch_size]
            for i in range(0, len(questions_with_ids), args.batch_size)
        ]:
            # Process all problems in batches
            ids, questions = zip(*batch)
            partial_solutions: List[str] = template(list(questions), problem_ids=list(ids))
            outputs.extend(partial_solutions)
        
        answer_messages = [
            [{
                "role": "user", 
                "content": (
                    f"Extract the final answer, making sure to obey the formatting instructions.\nSolution:\n{output}\n\n"
                    "Formatting instructions:\n- Final answer should be wrapped in \\boxed{{}}."
                )
            }] for output in outputs
        ]
        # get all outputs
        batch_completions = batch_completion(
            model=args.model,
            messages=answer_messages,
            temperature=args.temperature,
            stream=False,
            api_key=args.api_key,
            api_base=args.base_url,
        )
        outputs = [
            ori_output + "\n\n" + completion.choices[0].message.content 
            for ori_output, completion in zip(outputs, batch_completions)
        ]

    return outputs

def main(data_name, args):
    examples, output_dir = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    outputs = solve_problems(args, samples, output_dir)

    # extract preds
    results = [
        run_execute(executor, output, args.prompt_type, data_name) for output in outputs
    ]

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(outputs[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.update({"output": outputs[i], "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    for sample in all_samples:
        content = ""
        content += f"# Question\n\n{sample['question']}\n\n"
        content += f"# Output\n\n{sample['output']}\n\n"
        content += f"# Prediction\n\n{sample['pred']}\n\n"
        content += f"# GT COT\n\n{sample['gt_cot']}\n\n"
        content += f"# GT\n\n{sample['gt']}\n\n"
        template = Template(open("template.html").read()).render(content=content)

        with open(os.path.join(output_dir, f"{sample['idx']}.html"), "w") as f:
            f.write(template)

    with open(
        os.path.join(output_dir, f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
