import numpy as np
import string

# Letters contain the tokens, except for the blank token
letters = [' '] + list(string.ascii_uppercase)

def convert_text_to_array(text, start=1):
        """
        Compute the corresponding array given a text
        """
        array = []
        for character in list(text):
            array.append(letters.index(character) + start)
        return np.array(array)


def convert_array_to_text(array, start=1):
        """
        Compute the truth text given a truth array of indexes
        """
        text = []
        # Parse the array
        for idx in array:
            # Append only if not blank token (start = 1)
            if(idx >= start):
                text.append(letters[idx - start])     
        return ''.join(text).strip()

 
def ctc_convert_array_to_text(array_index_letters, start=1):
        """
        Compute the text prediction given a ctc output
        """
        # Previous letter
        previous_letter = -1
        outputed_text = []
        for n in array_index_letters:
            # Not same letters successively, and if n is lower than start it does not correspond to anything
            if(previous_letter != n and n >= start):
                # Append the corresponding letter
                outputed_text.append(letters[n - start]) 
            # Update previous letter              
            previous_letter = n
        return ''.join(outputed_text).strip()