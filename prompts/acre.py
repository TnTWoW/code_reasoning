example_prompt = """Input: {input}
Output: {output}"""


io_prompt = """Generate an output corresponding to the given input. Each example is an input-output pair. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on", "off", or "undetermined", indicating the state of the light or if the state of the light cannot be determined.

{examples}
Input: {test_input}
Output:"""


io_prompt_with_format = """Generate an output corresponding to the given input. Each example is an input-output pair. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on", "off", or "undetermined", indicating the state of the light or if the state of the light cannot be determined.

{examples}

Your input:
Input: {test_input}

Please format your output as follows:

Output: <Your output>"""


rule_prompt = """Generate a rule that maps the following inputs to their corresponding outputs. Each example is an input-output pair. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on" or "off", indicating the state of the light. For each object, determine whether it triggers the light to turn on, does not trigger it, or if it's undetermined.

{examples}

Please format your rule as follows:

Rule: {{"object 1": <"on"/"off"/"undetermined">, "object 2": <"on"/"off"/"undetermined">, ...}}"""


nl_rule_prompt = """Generate a rule that maps the following inputs to their corresponding outputs. Each example is an input-output pair. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on", "off", or "undetermined", indicating the state of the light or if the state of the light cannot be determined.

{examples}

Please format your rule as follows:

Rule: <Your rule>"""


feedback_prompt = """Input: {input}
Expected output: {target}
Actual output: {output}"""


rule_with_feedback_prompt = """Your rule: {rule}

Applying your rule to the following inputs does not produce the expected outputs.

{feedback}

Generate a new rule that maps the given inputs to their corresponding outputs. Please format your rule as follows:

Rule: {{"object 1": <"on"/"off"/"undetermined">, "object 2": <"on"/"off"/"undetermined">, ...}}"""


nl_rule_with_feedback_prompt = """Your rule: {rule}

Applying your rule to the following inputs does not produce the expected outputs.

{feedback}

Generate a new rule that maps the given inputs to their corresponding outputs. Please format your rule as follows:

Rule: <Your rule>"""


rule_to_output_prompt = """Generate an output corresponding to the given input based on the following rule. Each example is an input-output pair. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on", "off", or "undetermined", indicating the state of the light or if the state of the light cannot be determined. The rule indicates whether each object triggers the light to turn on, does not trigger it, or if it's undetermined.

Rule: {rule}

Input: {test_input}
Output:"""


rule_to_output_prompt_with_format = """Generate an output corresponding to the given input based on the following rule. Each example is an input-output pair. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on", "off", or "undetermined", indicating the state of the light or if the state of the light cannot be determined. The rule indicates whether each object triggers the light to turn on, does not trigger it, or if it's undetermined.

Rule: {rule}

Your input:
Input: {test_input}

Please format your output as follows:

Output: <Your output>"""


nl_rule_to_output_prompt = """Generate an output corresponding to the given input based on the following rule. Each example is an input-output pair. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on", "off", or "undetermined", indicating the state of the light or if the state of the light cannot be determined.

Rule: {rule}

Input: {test_input}
Output:"""


nl_rule_to_output_prompt_with_format = """Generate an output corresponding to the given input based on the following rule. Each example is an input-output pair. The input is a list of objects. The presence of certain objects will trigger the light to turn on. The output is either "on", "off", or "undetermined", indicating the state of the light or if the state of the light cannot be determined.

Rule: {rule}

Your input:
Input: {test_input}

Please format your output as follows:

Output: <Your output>"""
