import ast
import copy
import logging
import re
import os

import numpy as np
from tqdm import tqdm

from utils.format_utils import extract_response, flatten, unflatten
from utils.python_utils import execute_function, extract_program
from utils.query_utils import get_cost, query_batch
from utils.generate_utils import generate_batch

import transformers
import torch

logger = logging.getLogger(__name__)

PRINT_NUM = 3


class Task:
    def __init__(
        self,
        data,
        mode,
        model_name,
        method,
        n_train,
        n_test,
        n=1,
        temperature=0.0,
        max_iter=1,
        rule_type="default",
        interpreter_type="default",
        system_msg=None,
        history_file=None,
        cache_file=None,
        verbose=False,
        eval_every=-1,
        rules=None,
        interactions=None,
        cost=0,
        metrics=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.mode = mode
        self.method = method
        self.data = data
        self.n_train = n_train
        self.n_test = n_test
        self.n = n
        self.temperature = temperature
        self.max_iter = max_iter
        self.rule_type = rule_type
        self.interpreter_type = interpreter_type
        self.system_msg = system_msg
        self.history_file = history_file
        self.cache_file = cache_file
        self.verbose = verbose
        self.histories = [[] for _ in range(len(data))]
        self.interactions = (
            [[] for _ in range(len(data))] if interactions is None else interactions
        )
        self.rules = [[] for _ in range(len(data))] if rules is None else rules
        self.metrics = [] if metrics is None else metrics
        self.cost = cost
        self.eval_every = eval_every

        # We assume the following prompts are specified
        self.io_prompt = None
        self.example_prompt = None
        self.rule_prompt = None
        self.feedback_prompt = None
        self.rule_with_feedback_prompt = None
        self.rule_to_output_prompt = None
        self.cache = {}

    def get_train_examples(self, data):
        return data["train"][: self.n_train]

    def get_test_examples(self, data):
        return data["test"][: self.n_test]

    def get_all_examples(self, split, idxs=None):
        if idxs is None:
            idxs = list(range(len(self.data)))
        if split == "train":
            return [self.get_train_examples(self.data[i]) for i in idxs]
        elif split == "test":
            return [self.get_test_examples(self.data[i]) for i in idxs]
        else:
            raise ValueError(f"Invalid split: {split}")

    def format_input(self, input):
        return input

    def format_output(self, output):
        return output

    def format_examples(self, examples):
        example_strs = []
        for ex in examples:
            input = self.format_input(ex["input"])
            output = self.format_output(ex["output"])
            example_strs.append(self.example_prompt.format(input=input, output=output))
        return "\n".join(example_strs)

    def extract_prediction(self, text):
        pattern = r"Output:\s*(.*)"
        results = re.findall(pattern, text)
        if not results:
            return text
        return results[-1]

    def format_rule(self, rule):
        return rule

    def get_rule(self, response):
        prefixes = [
            "Updated Rule",
            "Updated rule",
            "Revised Rule",
            "Revised rule",
            "Rule",
        ]
        rule = extract_response(prefixes, response)
        return rule

    def get_histories(self, idxs):
        histories = []
        for i in idxs:
            history = self.histories[i]
            histories.append(history if len(history) > 0 else None)
        return histories

    def query(self, prompts, idxs, n=None, temperature=None, histories=None):
        n = self.n if n is None else n
        temperature = self.temperature if temperature is None else temperature
        responses = query_batch(
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
                # cost = get_cost(
                #     prompt, response, model_name=self.model_name, history=history
                # )
                # self.cost += cost
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

    def load_model(self):
        # if '8B' in self.model_name:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # if '70B' in self.model_name:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
        local_model = '/data/share_weight/' + self.model_name
        pipeline = transformers.pipeline(
            "text-generation",
            model=local_model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.pipeline = pipeline

    def generate(self, prompts, idxs, n=None, temperature=None, histories=None):
        n = self.n if n is None else n
        temperature = self.temperature if temperature is None else temperature
        responses = generate_batch(
            prompts,
            self.pipeline,
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

    def add_histories(self, role, idxs, prompts):
        for i, prompt in zip(idxs, prompts):
            self.histories[i].append({"role": role, "content": prompt})

    def add_rules(self, idxs, rules):
        for i, rule in zip(idxs, rules):
            self.rules[i].append(rule)

    def apply_all_rules(self, idxs, all_rules, all_examples):
        # idxs: [idx1, idx2]
        # all_rules: [rule1, rule2]
        # all_examples: [[ex1, ex2], [ex1, ex2]]
        raise NotImplementedError

    def apply_all_rules_with_lm(self, idxs, all_rules, all_examples):
        prompts = []
        prompt_idxs = []
        for idx, rule, examples in zip(idxs, all_rules, all_examples):
            rule = self.format_rule(rule)
            for example in examples:
                input = self.format_input(example["input"])
                prompt = self.rule_to_output_prompt.format(rule=rule, test_input=input)
                prompts.append(prompt)
                prompt_idxs.append(idx)
        if self.mode == 'generate':
            responses_flatten = self.generate(
                prompts, prompt_idxs, histories=None
            )
        else: # self.mode == 'query'
            responses_flatten = self.query(
                prompts, prompt_idxs, n=1, temperature=0, histories=None
            )
        outputs_flatten = [self.extract_prediction(res) for res in responses_flatten]
        outputs = unflatten(outputs_flatten, all_examples)
        return outputs

    def get_best_outputs(self, responses):
        if self.n == 1:
            return responses
        responses = [
            [str(self.extract_prediction(r)) for r in res] for res in responses
        ]
        best_outputs = [max(set(res), key=res.count) for res in responses]
        return best_outputs

    def get_best_responses(self, all_idxs, all_examples, all_responses):
        # all_idxs: [idx1, idx2]
        # all_examples: [[ex1, ex2], [ex1, ex2]]
        # responses [[rule1_for_examples1, rule2_for_examples1], [rule1_for_examples2, rule2_for_examples2]]
        assert len(all_examples) == len(all_responses) == len(all_idxs)
        # all_examples_flatten: [examples1, examples1, examples2, examples2]
        all_examples_flatten, all_responses_flatten = flatten(
            all_examples, all_responses
        )
        all_idxs_flatten, _ = flatten(all_idxs, all_responses)
        all_rules_flatten = [
            self.get_rule(response) for response in all_responses_flatten
        ]
        all_outputs_flatten = self.apply_all_rules(
            all_idxs_flatten,
            all_rules_flatten,
            all_examples_flatten,
        )
        accs = [
            self.eval_one(examples, outputs)[0]
            for examples, outputs in zip(all_examples_flatten, all_outputs_flatten)
        ]
        accs = unflatten(accs, all_responses)
        best_idx = [acc.index(max(acc)) for acc in accs]
        best_responses = [
            responses[idx] for idx, responses in zip(best_idx, all_responses)
        ]
        if self.verbose:
            for best_idx, best_response, acc in zip(best_idx, best_responses, accs):
                logger.info(f"Best rule: {best_response}, acc: {acc[best_idx]}")
        return best_responses

    def get_feedback(self, examples, outputs):
        feedbacks = []
        for ex, pred in zip(examples, outputs):
            target = self.format_output(ex["output"])
            input = self.format_input(ex["input"])
            pred = self.format_output(pred)
            if pred != target:
                feedback = self.feedback_prompt.format(
                    input=input, target=target, output=pred
                )
                feedbacks.append(feedback)
        return "\n".join(feedbacks)

    def eval_one(self, examples, outputs):
        # examples: [ex1, ex2]
        # outputs: [output1, output2]
        targets = [self.format_output(ex["output"]) for ex in examples]
        outputs = [self.format_output(output) for output in outputs]
        accs = []
        for pred, target in zip(outputs, targets):
            accs.append(float(pred == target))
        acc = np.mean(accs)
        return acc, accs

    def get_metrics(self, split, all_outputs):
        all_accs = []
        all_examples = self.get_all_examples(split)
        for examples, outputs in zip(all_examples, all_outputs):
            acc, accs = self.eval_one(examples, outputs)
            all_accs.append(accs)
        accs = np.array([np.mean(acc) for acc in all_accs])
        acc = np.mean(accs)
        instance_acc = (accs == 1).mean()
        output_dict = {
            "test_acc": float(acc),
            "test_instance_acc": float(instance_acc),
            "test_accs": all_accs,
        }
        return output_dict

    def eval_io(self):
        all_train_examples = self.get_all_examples("train")
        all_test_examples = self.get_all_examples("test")
        prompts = []
        idxs = []
        for idx, (train_examples, test_examples) in enumerate(
            zip(all_train_examples, all_test_examples)
        ):
            train_examples = self.format_examples(train_examples)
            for test_example in test_examples:
                test_input = self.format_input(test_example["input"])
                prompts.append(
                    self.io_prompt.format(
                        examples=train_examples, test_input=test_input
                    )
                )
                idxs.append(idx)
        responses = self.query(prompts, idxs, histories=None)
        responses = self.get_best_outputs(responses)
        responses = [self.extract_prediction(r) for r in responses]
        test_outputs = unflatten(responses, all_test_examples)
        metrics = self.get_metrics("test", test_outputs)
        self.metrics.append(metrics)

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
                responses = self.query(prompts, idxs, histories=histories)
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

    def eval_test_from_rule(self, responses):
        assert all([response is not None for response in responses])
        rules = [self.get_rule(response) for response in responses]
        all_test_examples = self.get_all_examples("test")
        idxs = list(range(len(all_test_examples)))
        logger.info(f"Applying rules to {len(all_test_examples)} test examples...")
        all_test_outputs = self.apply_all_rules(idxs, rules, all_test_examples)
        all_test_examples = self.get_all_examples("test")
        output_dict = self.get_metrics("test", all_test_outputs)
        return output_dict

    def eval_rule_application(self):
        metrics = {}
        rules = [rules[-1] for rules in self.rules]
        all_test_examples = self.get_all_examples("test")
        idxs = list(range(len(all_test_examples)))
        outputs = self.apply_all_rules_with_lm(idxs, rules, all_test_examples)
        rule_metrics = self.get_metrics("test", outputs)
        metrics["rule_metrics"] = rule_metrics
        test_acc = rule_metrics["test_acc"] * 100
        test_instance_acc = rule_metrics["test_instance_acc"] * 100
        logger.info("Query using rules:")
        logger.info(
            f"Accuracy: {test_acc:.2f}, instance accuracy: {test_instance_acc:.2f}"
        )
        return metrics

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

    def to_dict(self):
        output = {
            "model_name": self.model_name,
            "method": self.method,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n": self.n,
            "temperature": self.temperature,
            "max_iter": self.max_iter,
            "cost": self.cost,
            "metrics": self.metrics,
            "rule_type": self.rule_type,
            "interpreter_type": self.interpreter_type,
            "system_msg": self.system_msg,
            "rules": self.rules if hasattr(self, "rules") else None,
            "interactions": self.interactions,
        }
        return output


class PythonTask(Task):
    """Task that uses Python interpreter."""

    def get_python_input(self, input):
        return ast.literal_eval(input)

    def get_rule(self, response):
        if self.rule_type == "python":
            return extract_program(response)
        return super().get_rule(response)

    def format_rule(self, rule):
        if self.rule_type == "python":
            return rule
        return super().format_rule(rule)

    def apply_all_rules(self, idxs, all_rules, all_examples):
        if self.interpreter_type == "lm":
            return self.apply_all_rules_with_lm(idxs, all_rules, all_examples)
        if self.rule_type != "python":
            all_rules = self.rules_to_programs(idxs, all_rules)
            programs = [extract_program(rule) for rule in all_rules]
        else:
            programs = all_rules
        all_outputs = []

        total = len(all_examples)
        for examples, program in tqdm(
            zip(all_examples, programs), desc="Applying rules", total=total
        ):
            inputs = copy.deepcopy(
                [self.get_python_input(ex["input"]) for ex in examples]
            )
            outputs = execute_function(program, inputs)
            all_outputs.append(outputs)
        return all_outputs

    def rules_to_programs(self, idxs, all_rules):
        prompts = [self.rule_to_python_prompt.format(rule=rule) for rule in all_rules]
        if self.mode == "generate":
            responses = self.generate(prompts, idxs, histories=None)
        else:
            responses = self.query(prompts, idxs, n=1, temperature=0, histories=None)
        return responses
