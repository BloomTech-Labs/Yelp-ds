import numpy as np
import copy

class SequenceCandidate(object):
    
    @staticmethod
    def template_seq(start_idx = 1, max_length = 15, ignore_idx = None, alpha = .9):
        seq = np.repeat(0,15)
        seq[0] = start_idx
        
        probs = np.repeat(0.0,15)
        probs[0] = 1
        return SequenceCandidate(seq, probs, max_length, ignore_idx, alpha)
        
        
    
    def __init__(self, seq, probs, max_length = 15, ignore_idx = None, alpha = .9):
        assert len(seq) == max_length
        self._max_length = max_length
        self._seq = seq
        self._probs = probs
        
        self._num_elem = max_length 
        for i in range(len(seq)):
            if seq[i] == 0:
                self._num_elem = i  
                break
        
        self._bigrams = set()
        self._ignore_idx = ignore_idx
        if ignore_idx is None:
            self._ignore_idx = []
        self._prob_weights = [alpha**i for i in range(max_length)]
    
    
    def add_token(self, token, prob):
        
        if self._num_elem >= self._max_length:
            raise IndexError("Sequence is already populated.\nCan't add any more tokens to it.")
        
        newcandidate = copy.deepcopy(self)
        
        newcandidate._seq[self._num_elem] = token
        
        newcandidate._probs[self._num_elem] = prob
        
        newcandidate._bigrams.add(tuple(newcandidate._seq[self._num_elem - 1 : newcandidate._num_elem + 1]))
        
        newcandidate._num_elem += 1
        return(newcandidate)
    
    def probsum(self):
        
        valid_probs = self._probs[~np.in1d(self._seq, self._ignore_idx)]
        
        return np.sum(np.multiply(valid_probs, self._prob_weights[:len(valid_probs)]))
    
    def final_token(self):
        return self._seq[self._num_elem - 1]
    
     
    def to_words(self,reverse_tokenizer, end_idx):
        
        out_words = []
        for i in range(1,len(self._seq)):
            
            idx = self._seq[i]
            if idx == 0 or idx == end_idx:
                break
            if idx in self._ignore_idx:
                continue
            
            if self._seq[i - 1] != idx:
                out_words.append(reverse_tokenizer[idx])
        out_string = " ".join(out_words)
        return out_string
    
    
    def __lt__(self, other):
        try:
            return self.probsum() < other.probsum()
        except AttributeError: 
            return NotImplemented