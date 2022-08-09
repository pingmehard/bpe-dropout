import sentencepiece as spm

class spm_encoder:

    def __init__(self, vocab_size = None):

        if vocab_size == None:
            self.vocab_size = 2000
        else:
            self.vocab_size = vocab_size

    def spm_encode(self, string, out_type, dropout):
        '''Coding string to BPE-vector'''
        if dropout:
            # If you'd like to add dropout to your BPE tokens.
            return self.sp.encode(string, nbest_size=-1, out_type=out_type, enable_sampling=True, alpha=0.1)
        else:
            # Return tokens without dropout.
            return self.sp.encode(string, nbest_size=-1, out_type=out_type)

    def train_model(self, iterable_strings):
        '''Train bpe model with any iterable strigns on entry.'''

        # SPM requires using file locally or remotely, but this app requires just list of strings.
        with open('spm_texts.txt', 'w', encoding="utf-8") as f:
            for i in iterable_strings:
                f.write(i + '\n')
        print('File spm_texts.txt for model created.')

        try:
            # Train bpe model.
            spm.SentencePieceTrainer.train(input='spm_texts.txt', model_prefix='m', vocab_size=self.vocab_size, model_type='bpe')
            print('Model trained!')

        except Exception as exx:
            # Check if we have limit for bpe vocab. And chenges the limit to suggestion from SPT.
            if 'too high' in str(exx):
                self.vocab_size = int(str(exx).split()[-1].replace('.',''))
                print(f'Vocab size set to {self.vocab_size}')
            spm.SentencePieceTrainer.train(input='spm_texts.txt', model_prefix='m', vocab_size=self.vocab_size, model_type='bpe')
            print('Model trained!')

        self.sp = spm.SentencePieceProcessor(model_file='m.model')
        print('Model initialized!')

        return self.sp

    def transform(self, iterable_strings, out_type = int, dropout = False):
        '''Transform your strings to BPE vector.
        Give:
        out_type == int or str. (Example: out_type = str) By default it is int.
        dropout == False or True. (Example: dropout = True) By default it is False.
        '''
        result = []
        for str in iterable_strings:
            result.append(self.spm_encode(str, dropout = dropout, out_type = out_type))

        return result

    def load_model(self, path_to_model):
        '''Load trained model. Requires path to .model file.'''
        self.sp = spm.SentencePieceProcessor(model_file=path_to_model)

        return self