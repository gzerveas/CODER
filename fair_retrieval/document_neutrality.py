import collections
import numpy as np
import pickle
import pdb

class DocumentNeutrality:
    
    def __init__(self, representative_words_path, threshold=1, groups_portion={'f':0.5, 'm':0.5}):
        
        self.representative_words_path = representative_words_path
        self.threshold = threshold
        self.groups_portion = groups_portion
        
        self.representative_words = {}
        for _group in self.groups_portion:
            self.representative_words[_group] = []

        for l in open(self.representative_words_path):
            vals = l.strip().split(',')
            _word = vals[0].lower()
            _group = vals[1]
            
            if _group not in self.groups_portion:
                raise Exception("Group %s is observed in representative words but is not defined in groups_portion" % _group)
            
            self.representative_words[_group].append(_word)

        for _group in self.representative_words:
            self.representative_words[_group] = set(self.representative_words[_group])

    def get_magnitude_count(self, tokens):
        _text_cnt = collections.Counter(tokens)

        _group_magnitudes = {}
        for _group in self.groups_portion:
            _group_magnitudes[_group] = 0
            
        for _word in _text_cnt:
            for _group in self.groups_portion:
                if _word in self.representative_words[_group]:
                    _group_magnitudes[_group] += _text_cnt[_word]
            
        return _group_magnitudes

    def get_neutrality(self, tokens):
        _group_magnitudes = self.get_magnitude_count(tokens)
        _group_magnitudes_sum = np.sum(list(_group_magnitudes.values()))
        
        _neutrality = 1
        if _group_magnitudes_sum > self.threshold:
            for _group in self.groups_portion:
                _distribution = _group_magnitudes[_group] / float(_group_magnitudes_sum)
                _neutrality -= np.abs(_distribution - self.groups_portion[_group])

        return _neutrality
    
