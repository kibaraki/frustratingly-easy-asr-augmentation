import json

def analyze_json_file(file_path):

    alts = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Count the number of unique keys
        num_keys = len(data.keys())

        # Use a set to store unique words 
        unique_words = set()

        # Iterate through the lists of words in the json values
        for word_list in data.values():
            # Add all words from the list to the set
            if len(word_list) > 1:
                alts += 1
            unique_words.update(word_list)

        # Count the total number of unique words
        num_unique_words = len(unique_words)

        return num_keys, num_unique_words, alts

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None, None, None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None, None, None



#file_name = '../nashta/nash_gloss.json'#'../vatlongos/vat_gloss.json'
file_name = '../librispeech/libri-100_gloss.json' # './vat_gloss.json'
keys_count, words_count, alts = analyze_json_file(file_name)

if keys_count is not None:
    print(f"Analysis for '{file_name}':")
    print(f"Number of unique gloss tags (keys): {keys_count}")
    print(f"Number of unique words (in the sets): {words_count}")
    print(f"Number of glosses with alternatives: {alts}")
