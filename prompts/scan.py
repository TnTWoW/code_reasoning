example_prompt = """Input: {input}
Output: {output}"""


io_prompt = """Generate an output corresponding to the given input based on the following examples.

{examples}
Input: {test_input}
Output:"""


io_prompt_with_format = """Generate an output corresponding to the given input based on the following examples.

{examples}

Your input:
Input: {test_input}

Please format your output as follows:

Output: <Your output>"""


rule_prompt = """Generate grammar rules that map the following inputs to their corresponding outputs. Your grammar rules should follow the format "<input> -> <output>". Use the prefix "##" to denote a nonterminal symbol. For instance, "##A twice -> ##A ##A". The left-hand side cannot contain repetitive nonterminal symbols; i.e., rules like "##A ##A -> ##A twice" or "##A and ##A -> ##A twice" are not allowed. Ensure that the number of unique nonterminal symbols on the left-hand side matches that on the right-hand side in your rules. For each rule, assign an integer as its priority. A higher priority indicates that the rule should be considered first when generating parses. Try to make your rules as minimal as possible.

{examples}

Please format your rules as follows:

Rule 1: <Your rule>
Priority 1: <Your priority>
..."""


nl_rule_prompt = """Generate rules that map the following inputs to their corresponding outputs. Specify the priority of the rules if necessary. Try to make your rules as minimal and generally applicable as possible.

{examples}

Please format your rules as follows:

Rule 1: <Your rule>
..."""


feedback_prompt = """Input: {input}
Expected output: {target}
Actual output: {output}"""


rule_with_feedback_prompt = """Your rules: 
{rule}

These rules do not work for the following examples.

{feedback}

Generate new rules that map the given inputs to their corresponding outputs. Please format your rule as follows:

Rule 1: <Your rule>
..."""


rule_to_output_prompt = """Generate an output corresponding to the given input based on the following grammar rules. The grammar rules follow the format "<input> -> <output>". The "##" prefix denotes a nonterminal symbol. For instance, ##A twice -> ##A ##A. Each rule has an associated priority. A higher priority indicates that the rule should be considered first when generating parses. The output is a sequence of tokens joined by spaces.

Rules:
{rule}

Input: {test_input}
Output:"""


rule_to_output_prompt_with_format = """Generate an output corresponding to the given input based on the following grammar rules. The grammar rules follow the format "<input> -> <output>". The "##" prefix denotes a nonterminal symbol. For instance, ##A twice -> ##A ##A. Each rule has an associated priority. A higher priority indicates that the rule should be considered first when generating parses. The output is a sequence of tokens joined by spaces.

Rules:
{rule}

Your input:
Input: {test_input}

Please format your output as follows:

Output: <Your output>"""


nl_rule_to_output_prompt = """Generate an output corresponding to the given input based on the following rules. The output is a sequence of tokens joined by spaces.

Rules:
{rule}

Input: {test_input}
Output:"""


nl_rule_to_output_prompt_with_format = """Generate an output corresponding to the given input based on the following rules. The output is a sequence of tokens joined by spaces.

Rules:
{rule}

Your input:
Input: {test_input}

Please format your output as follows:

Output: <Your output>"""
