import argparse
import logging
import os

# from tasks.acre import ACRE
from tasks.arc import ARC
from tasks.list_function import ListFunction
from tasks.robust_fill import RobustFill
# from tasks.scan import SCAN
from utils.io_utils import read_jsonl, write_json
from utils.query_utils import CACHE_FILE, HISTORY_FILE


os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the data file.",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="query",
        choices=["generate", "query"],
        help="Set the mode to generate or query.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of the task.",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="Number of examples to train.",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=None,
        help="Number of examples to test.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of examples to sample.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1,
        help="Number of iterations",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rule",
        help="Method to use",
    )
    parser.add_argument(
        "--rule_type",
        type=str,
        default="default",
        help="Rule type to use",
    )
    parser.add_argument(
        "--interpreter_type",
        type=str,
        default="default",
        help="Interpreter type to use",
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default=None,
        help="Path to the cache file.",
    )
    parser.add_argument(
        "--history_file",
        type=str,
        default=None,
        help="Path to the history file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print out the intermediate results.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=-1,
        help="Evaluate every n iterations.",
    )
    return parser.parse_args()


NAME_TO_TASK = {
    "list_function": ListFunction,
    "arc": ARC,
    "robust_fill": RobustFill,
    # "acre": ACRE,
    # "scan": SCAN,
}


class MessageFilter(logging.Filter):
    def filter(self, record):
        return (
            "error_code=rate_limit_exceeded" in record.getMessage()
            or "response_code=200" in record.getMessage()
            or "429 Too Many Requests" in record.getMessage()
        )


def main():
    args = parse_args()
    openai_logger = logging.getLogger("openai")
    openai_logger.addFilter(MessageFilter())
    openai_logger.setLevel(logging.WARNING)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.addFilter(MessageFilter())
    httpx_logger.setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s:%(lineno)d - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if args.output_file is not None:
        assert not os.path.exists(args.output_file)
        dirname = os.path.dirname(args.output_file)
    else:
        dirname = os.getcwd()
    if args.cache_file is None:
        args.cache_file = os.path.join(dirname, CACHE_FILE)
    if args.history_file is None:
        args.history_file = os.path.join(dirname, HISTORY_FILE)
    data = read_jsonl(args.data_file)
    if args.n_examples is not None:
        data = data[: args.n_examples]
    task = NAME_TO_TASK[args.task_name](
        data=data,
        mode=args.mode,
        model_name=args.model_name,
        method=args.method,
        n_train=args.n_train,
        n_test=args.n_test,
        n=args.n_samples,
        temperature=args.temperature,
        max_iter=args.max_iter,
        rule_type=args.rule_type,
        interpreter_type=args.interpreter_type,
        cache_file=args.cache_file,
        history_file=args.history_file,
        verbose=args.verbose,
        eval_every=args.eval_every,
    )
    output_dict = task.run()

    logger.info(f"Total cost: {task.cost}")

    if args.output_file is not None:
        write_json(output_dict, args.output_file)
        logger.info(f"Output file saved to {args.output_file}")


if __name__ == "__main__":
    main()
