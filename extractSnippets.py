def extract_snippets(input_file, output_file, num_snippets=200):
    with open(input_file, 'r') as file:
        data = file.read()

    snippets = data.split('-----')
    snippets = [snippet.strip() for snippet in snippets if snippet.strip()]

    if len(snippets) < num_snippets:
        print(f"Warning: The input file contains fewer than {num_snippets} snippets.")
        num_snippets = len(snippets)

    selected_snippets = snippets[:num_snippets]

    with open(output_file, 'w') as file:
        for snippet in selected_snippets:
            file.write(snippet + '\n-----\n')

    print(f"{num_snippets} snippets saved to {output_file}")

# Specify the input and output files
input_file = 'invalid_dc_fake2.txt'
output_file = 'invalid_dc_testing_fake2.txt'

# Extract and save snippets
extract_snippets(input_file, output_file)
