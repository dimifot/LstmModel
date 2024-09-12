import random

def read_snippets(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        snippets = content.split("-----")
        # Filter out any empty strings that may occur from splitting
        snippets = [snippet.strip() for snippet in snippets if snippet.strip()]
    return snippets

def write_snippets(file_path, snippets):
    with open(file_path, 'w') as file:
        for snippet in snippets:
            file.write(snippet.strip())
            file.write("\n-----\n")

def delete_snippets(file_path, num_to_delete):
    snippets = read_snippets(file_path)
    if num_to_delete > len(snippets):
        print(f"Number to delete ({num_to_delete}) is greater than the number of snippets ({len(snippets)}).")
        return
    snippets_to_keep = random.sample(snippets, len(snippets) - num_to_delete)
    write_snippets(file_path, snippets_to_keep)

if __name__ == "__main__":
    num_to_delete = 896 # Specify the number of snippets to delete

    #delete_snippets('valid_snippets.txt', num_to_delete)
    delete_snippets('invalid_py_fake.txt', num_to_delete)

    print(f"Deleted {num_to_delete} snippets from each file.")