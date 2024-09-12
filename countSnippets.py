with open('invalid_dc_new.txt', 'r') as file:
    test_snippets = file.read().strip().split('-----')

print(len(test_snippets))
