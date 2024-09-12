import random
import re
import os
import string

def introduce_grammar_errors(code_snippet):
    def remove_colon(code):
        return re.sub(r':', '', code, count=1)

    def remove_semicolon(code):
        return re.sub(r';', '', code, count=1)

    def remove_right_parenthesis(code):
        return re.sub(r'[)]', '', code, count=1)

    def remove_left_parenthesis(code):
        return re.sub(r'[(]', '', code, count=1)

    def remove_right_brace(code):
        return re.sub(r'[}]', '', code, count=1)

    def remove_left_brace(code):
        return re.sub(r'[{]', '', code, count=1)

    def wrong_if_else(code):
        if "if" in code and "else" in code:
            code = re.sub(r'else', 'if', code, count=1)
        return code

    errors = [
        remove_colon, remove_semicolon, remove_right_parenthesis,
        remove_left_parenthesis, remove_right_brace, remove_left_brace,
        wrong_if_else
    ]

    num_errors = random.randint(1, 4)  # Random number of errors between 1 and 10
    print(f"Applying {num_errors} errors to the snippet")
    for _ in range(num_errors):
        error = random.choice(errors)
        code_snippet = error(code_snippet)
    return code_snippet


def missing_var_reference_initialization_only(code):
    pattern = re.compile(r'\bfloat (\w+) = [^;]+;')
    matches = pattern.findall(code)

    if matches:
        var = random.choice(matches)
        new_var = f'{generate_random_string()}'
        counter = 1
        while re.search(fr'\b{new_var}\b', code):
            new_var = f'{generate_random_string()}{counter}'
            counter += 1
        code = re.sub(fr'\bfloat {var} =', f'float {new_var} =', code)
    return code

def missing_var_reference_usage_only(code):
    pattern = re.compile(r'\bfloat (\w+) = [^;]+;')
    matches = pattern.findall(code)

    if matches:
        var = random.choice(matches)
        new_var = f'{generate_random_string()}'
        counter = 1
        while re.search(fr'\b{new_var}\b', code):
            new_var = f'{generate_random_string()}{counter}'
            counter += 1
        code = re.sub(fr'(?<!float )\b{var}\b', new_var, code)
    return code

def missing_method_reference_initialization_only(code):
    pattern = re.compile(r'\bdef (\w+)\(')
    matches = pattern.findall(code)

    if matches:
        method = random.choice(matches)
        new_method = f'{generate_random_string()}'
        counter = 1
        while re.search(fr'\b{new_method}\b', code):
            new_method = f'{generate_random_string()}{counter}'
            counter += 1
        code = re.sub(fr'\bdef {method}\(', f'def {new_method}(', code)
    return code

def missing_method_reference_usage_only(code):
    pattern = re.compile(r'\bdef (\w+)\(')
    matches = pattern.findall(code)

    if matches:
        method = random.choice(matches)
        new_method = f'{generate_random_string()}'
        counter = 1
        while re.search(fr'\b{new_method}\b', code):
            new_method = f'{generate_random_string()}{counter}'
            counter += 1
        code = re.sub(fr'\b{method}\(', f'{new_method}(', code)
    return code

def generate_random_string():
    length = random.randint(3, 7)
    random_string = ''.join(random.choices(string.ascii_lowercase, k=length))
    return random_string

actions = [introduce_grammar_errors, missing_var_reference_initialization_only, missing_var_reference_usage_only, missing_method_reference_initialization_only, missing_method_reference_usage_only]

def process_file(input_file):
    with open(input_file, 'r') as file:
        data = file.read()

    print("Input file read successfully.")
    snippets = data.split('-----')
    invalid_snippets = {action.__name__: [] for action in actions}

    for snippet in snippets:
        snippet = snippet.strip()
        if snippet:
            print(f"Processing snippet:\n{snippet}\n")
            selected_action = random.choice(actions)
            invalid_snippet = selected_action(snippet)
            print(f"Invalid snippet generated:\n{invalid_snippet}\n")
            invalid_snippets[selected_action.__name__].append(invalid_snippet)

    #for action_name, snippets in invalid_snippets.items():
        #output_file = f'{action_name}.txt'
    output_file = f'invalid_dc_new.txt'
    with open(output_file, 'w') as file:
        for snippet in snippets:
            file.write(snippet + '\n-----\n')
    print(f"Invalid snippets saved to {output_file}")

# Specify the input file
input_file = 'valid_dc_new.txt'

# Process the input file
process_file(input_file)