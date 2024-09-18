import logging
import re
import copy

from prompts.arc import (
    example_prompt,
    feedback_prompt,
    io_prompt,
    io_prompt_with_format,
    python_rule_prompt,
    rule_prompt,
    structured_rule_prompt,
    rule_to_output_prompt,
    rule_to_output_prompt_with_format,
    rule_to_python_prompt,
    rule_with_feedback_prompt,
)
from tasks.base import PythonTask
from utils.query_utils import get_cost, query_batch_struct
from utils.format_utils import format_grid, format_list, str_to_list, unformat_grid
from utils.query_utils import CLAUDE_MODELS

logger = logging.getLogger(__name__)
PRINT_NUM = 3

class ARC(PythonTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.model_name in CLAUDE_MODELS:
            self.io_prompt = io_prompt_with_format
            self.rule_to_output_prompt = rule_to_output_prompt_with_format
        else:
            self.io_prompt = io_prompt
            self.rule_to_output_prompt = rule_to_output_prompt

        if self.rule_type == "python":
            self.rule_prompt = python_rule_prompt
        else:
            self.rule_prompt = structured_rule_prompt
        self.example_prompt = example_prompt
        self.rule_with_feedback_prompt = rule_with_feedback_prompt
        self.feedback_prompt = feedback_prompt
        self.rule_to_python_prompt = rule_to_python_prompt

    def struct_query(self, prompts, idxs, n=None, temperature=None, histories=None):
        n = self.n if n is None else n
        temperature = self.temperature if temperature is None else temperature
        responses = query_batch_struct(
            prompts,
            self.model_name,
            system_msg=self.system_msg,
            cache_file=self.cache_file,
            history_file=self.history_file,
            histories=histories,
            n=n,
            temperature=temperature,
        )
        prompt2key = lambda p, h: (
            p,
            self.model_name,
            self.system_msg,
            tuple([tuple(e.items()) for e in h]) if h is not None else None,
            temperature,
            n,
        )
        assert len(idxs) == len(prompts) == len(responses)
        for i, (idx, prompt, response) in enumerate(zip(idxs, prompts, responses)):
            history = histories[i] if histories is not None else None
            key = prompt2key(prompt, history)
            if key not in self.cache:
                self.cache[key] = response
                cost = get_cost(
                    prompt, response, model_name=self.model_name, history=history
                )
                self.cost += cost
            self.interactions[idx].append(
                {
                    "query": prompt,
                    "response": response,
                    "history": copy.deepcopy(history),
                    "n": n,
                    "temperature": temperature,
                    "system_msg": self.system_msg,
                }
            )
        return responses

    def canonicalize(self, grid):
        try:
            if isinstance(grid, list) and isinstance(grid[0], list):
                return [[int(x) if x is not None else 0 for x in row] for row in grid]
            elif isinstance(grid, list):
                return [int(x) for x in grid]
            elif isinstance(grid, str):
                return str_to_list(grid)
            return grid
        except Exception:
            return grid

    def format_input(self, input):
        return str(input)

    def format_output(self, output):
        return str(output)

    # def format_input(self, input):
    #     return "\n" + format_grid(input, row_sep="\n", sep=", ")
    #
    # def format_output(self, output):
    #     grid = self.canonicalize(output)
    #     try:
    #         if isinstance(grid, list) and isinstance(grid[0], list):
    #             return "\n" + format_grid(grid, row_sep="\n", sep=", ")
    #         elif isinstance(grid, list):
    #             return "\n" + format_list(grid, sep=", ", bracket=True)
    #         return "\n" + str(grid)
    #     except:
    #         return "\n" + str(grid)

    def get_rule(self, response):
        result = []
        json_response = eval(response)
        for i, step in enumerate(json_response['steps'], start=1):
            subrule = step['Subrule']
            input_data = step['input']
            output_data = step['output']
            result.append(f"Step {i}: {subrule} Input: {input_data}, Output: {output_data}")

        # Append the rule at the end
        result.append(f"Rule: {json_response['rule']}")

        # Join the list into a single string
        final_output = '. '.join(result)
        return final_output

    def eval_rule(self):
        prompts = []
        all_train_examples = self.get_all_examples("train")
        for train_examples in all_train_examples:
            train_examples = self.format_examples(train_examples)
            prompts.append(self.rule_prompt.format(examples=train_examples))
        idxs = list(range(len(self.data)))
        idx_to_response = [None for _ in range(len(self.data))]

        if self.mode == "generate":
            self.load_model()

        for i in range(self.max_iter):
            logger.info(
                f"======= Iteration {i}: query {len(prompts)} examples =========="
            )
            histories = self.get_histories(idxs)
            assert len(histories) == len(idxs)
            if self.mode == "generate":
                responses = self.generate(prompts, idxs, histories=histories)
            else:
                responses = self.struct_query(prompts, idxs, histories=histories)
            if self.n > 1:
                all_train_examples = self.get_all_examples("train", idxs)
                logger.info(f"Reranking {len(all_train_examples)} train examples...")
                if self.verbose:
                    logger.info(f"Responses before reranking:")
                    for res in responses[:PRINT_NUM]:
                        logger.info(res)
                responses = self.get_best_responses(idxs, all_train_examples, responses)
            for idx, response in zip(idxs, responses):
                idx_to_response[idx] = response

            rules = [self.get_rule(response) for response in responses]
            self.add_rules(idxs, rules)

            if self.eval_every > 0 and i % self.eval_every == 0:
                metrics = self.eval_test_from_rule(idx_to_response)
                self.metrics.append(metrics)

            if self.max_iter > 1:
                all_train_examples = self.get_all_examples("train", idxs)
                logger.info(
                    f"Applying rules to {len(all_train_examples)} train examples for feedback..."
                )
                if self.verbose:
                    logger.info(f"Rules:")
                    for rule in rules[:3]:
                        logger.info(rule)
                all_train_outputs = self.apply_all_rules(
                    idxs, rules, all_train_examples
                )
                self.add_histories("user", idxs, prompts)
                self.add_histories("assistant", idxs, responses)

                prompts = []
                new_idxs = []
                for idx, rule, train_examples, train_outputs in zip(
                    idxs, rules, all_train_examples, all_train_outputs
                ):
                    feedback = self.get_feedback(train_examples, train_outputs)
                    rule = self.format_rule(rule)
                    if self.verbose:
                        logger.info(f"Feedback:")
                        logger.info(feedback)
                    if feedback == "":
                        continue
                    train_examples = self.format_examples(train_examples)
                    prompt = self.rule_with_feedback_prompt.format(
                        examples=train_examples, rule=rule, feedback=feedback
                    )
                    prompts.append(prompt)
                    new_idxs.append(idx)
                idxs = new_idxs

                if len(prompts) == 0:
                    logger.info(f"No more feedback, break at iteration {i}")
                    break

        if self.eval_every <= 0:
            metrics = self.eval_test_from_rule(idx_to_response)
            self.metrics.append(metrics)

    def extract_prediction(self, text):
        pattern = r"Output:\s*(.*)"
        results = re.findall(
            pattern, text, re.DOTALL
        )  # re.DOTALL allows . to match newlines
        if not results:
            return text
        grid = results[-1]
        x = unformat_grid(grid, row_sep="\n", sep=", ")
        return x

    def get_python_input(self, input):
        return input
    
    def run(self):
        if self.method == "io":
            self.eval_io()
        else:
            self.eval_rule()

        metrics = self.metrics[-1]
        acc = metrics["test_acc"] * 100
        instance_acc = metrics["test_instance_acc"] * 100
        outputs = self.to_dict()
        logger.info(f"Mean accuracy: {acc:.2f}, instance accuracy: {instance_acc:.2f}")
        logger.info(f"Total cost: {self.cost}")
        return outputs
