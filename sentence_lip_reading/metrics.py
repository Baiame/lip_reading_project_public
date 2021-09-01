import editdistance

def compute_wer(predict, truth):
    """
    Return the WER given two strings of words
    """
    # Compute word alignment        
    pairs = [(pair[0].split(' '), pair[1].split(' ')) for pair in zip(predict, truth)]
    # Compute the WER with the distance between the words
    wer = [1.0*editdistance.eval(pair[0], pair[1])/len(pair[1]) for pair in pairs]
    return wer

    
def compute_cer(predict, truth):   
    """
    Return the CER given the array of predicted senteces and the array of true sentences
    """
    # Pair the sentences 2 by 2    
    zipped = zip(predict, truth)
    # Compute the CER with the distance between the 2 sentences
    cer = [1.0*editdistance.eval(sentence[0], sentence[1])/len(sentence[1]) for sentence in zipped]
    return cer