import argparse
import logging
import os

from tasks.acre import ACRE
from tasks.arc import ARC
from tasks.list_function import ListFunction
from tasks.scan import SCAN
from utils.io_utils import read_json, read_jsonl, write_json
from utils.query_utils import CACHE_FILE, HISTORY_FILE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the data file.",
    )
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        default=None,
        help="Path to the input file.",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Path to the output file.",
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
        "--n_examples",
        type=int,
        default=None,
        help="Number of examples to evaluate",
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
    return parser.parse_args()


NAME_TO_TASK = {
    "list_function": ListFunction,
    "arc": ARC,
    "acre": ACRE,
    "scan": SCAN,
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

    assert args.input_file is not None
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
    output_dict = read_json(args.input_file)
    if args.model_name:
        output_dict["model_name"] = args.model_name
    task = NAME_TO_TASK[args.task_name](
        data=data,
        cache_file=args.cache_file,
        history_file=args.history_file,
        verbose=args.verbose,
        **output_dict,
    )
    output_dict = task.to_dict()
    output_dict.update(task.eval_rule_application())
    logger.info(f"Total cost: {task.cost}")

    if args.output_file is not None:
        write_json(output_dict, args.output_file)
        logger.info(f"Output file saved to {args.output_file}")


if __name__ == "__main__":
    main()
