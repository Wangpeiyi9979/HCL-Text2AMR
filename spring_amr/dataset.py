import logging
import random
import torch
import penman
import numpy as np
from cached_property import cached_property
from torch.utils.data import Dataset
from spring_amr.IO import read_raw_amr_data
from my_amr.amr_utils import get_given_deep_amr, get_deep_amr

class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        self.amr_all_pointer = self.tokenizer.encoder['Ġparse-to-all-layer']
        for g in graphs:
            l, e = self.tokenizer.linearize(g)
            l[0] =  self.amr_all_pointer
            e['linearized_graphs'][0] = f'Ġparse-to-all-layer'
            try:
                self.tokenizer.batch_encode_sentences([g.metadata['snt']])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                print('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(g.metadata['snt'] + f" parse to all layer")
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)
        print(f'Data Num :{len(self.sentences)}')

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra

class AMRLayerDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
    ):
        self.init_token = 'Ġ'
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        deep_one_pointer_id = self.tokenizer.encoder['Ġparse-to-one-layer']
        for g in graphs:
            l, e = self.tokenizer.linearize(g)
            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            lines = penman.encode(g).split('\n')
            lines = list(filter(lambda x: not x.startswith('# ::'), lines))
            amr = "\n".join(lines)
            deepth = get_deep_amr(amr)
            for dp in range(1, deepth-2):
                amr_deep_dp = get_given_deep_amr(amr, dp)
                g_deep_dp = penman.decode(amr_deep_dp)
                try:
                    l_dp, e_dp = self.tokenizer.linearize(g_deep_dp)
                except:
                    print(amr_deep_dp)
                dp_pointer = deep_one_pointer_id + dp - 1
                l_dp[0] = dp_pointer
                e_dp['linearized_graphs'][0] = f'Ġparse-to-{dp}-layer'
                self.sentences.append(g.metadata['snt'] + f" parse to {dp} layer")
                self.graphs.append(g_deep_dp)
                self.linearized.append(l_dp)
                self.linearized_extra.append(e_dp)
        layer_ids = np.array([x[0] for x in self.linearized])
        layer_order = layer_ids.argsort()
        print("[AMR Data NUM]:{}".format(len(self.sentences)))
        print('[Sample]')
        import random
        idx = random.randint(1, len(self.sentences))
        print('src:  ' + self.sentences[idx])
        print('tgt: ' + self.tokenizer.decode(self.linearized[idx]))

    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra
    
class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

class CurriculumAMRDatasetTokenBatcherAndLoader:

    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False, IC_steps=500):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort
        self.IC_steps = IC_steps
        self.diffcults = self.get_diffcults()
        print(f'[MAX_layer] {self.diffcults.max()}; [IC_steps]: {IC_steps}')

    def __iter__(self):
        it = self.sampler_curriculum()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    def get_diffcults(self):
        diffcults = []
        for data in self.dataset:
            lines = penman.encode(data['graphs']).split('\n')
            lines = list(filter(lambda x: not x.startswith('# ::'), lines))
            amr = "\n".join(lines)
            deep = get_deep_amr(amr)            
            diffcults.append(deep)
        diffcults = np.array(diffcults) - min(diffcults)
        return diffcults

    def sampler_curriculum(self):
        max_layer = self.diffcults.max()
        for layer in range(max_layer+1):
            for _ in range(self.IC_steps):
                mask = self.diffcults <= layer
                optional_indexs = np.array(range(0, len(self.dataset)))[mask]
                ids = np.random.choice(optional_indexs, replace=False, size=100).tolist() # 选100个句子作为该batch候选

                batch_longest = 0
                batch_nexamps = 0 
                batch_ntokens = 0
                batch_ids = []

                def discharge():
                    nonlocal batch_longest
                    nonlocal batch_nexamps
                    nonlocal batch_ntokens
                    ret = batch_ids.copy()
                    batch_longest *= 0
                    batch_nexamps *= 0
                    batch_ntokens *= 0
                    batch_ids[:] = []
                    return ret
                while batch_ntokens < self.batch_size and len(ids) > 0:
                    idx = ids.pop()
                    size = self.dataset.size(self.dataset[idx])
                    cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
                    if cand_batch_ntokens > self.batch_size and batch_ids:
                        break
                    batch_longest = max(batch_longest, size)
                    batch_nexamps += 1
                    batch_ntokens = batch_longest * batch_nexamps
                    batch_ids.append(idx)
                yield discharge()

class CurriculumLayerDatasetTokenBatcherAndLoader:

    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, SC_steps=1000):
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.SC_steps = SC_steps 
        self.diffcults = self.get_diffcults()
        print(f'[MAX_layer] {self.diffcults.max()}; [SC steps]: {SC_steps}')

    def __iter__(self):
        it = self.sampler_curriculum()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    def get_diffcults(self):
        diffcults = []
        for l in self.dataset.linearized:
            diffcults.append(l[0])
        diffcults = np.array(diffcults) - min(diffcults)
        return diffcults


    def sampler_curriculum(self):
        max_layer = self.diffcults.max()
        for layer in range(max_layer+1):
            for _ in range(self.SC_steps):
                mask = self.diffcults <= layer
                optional_indexs = np.array(range(0, len(self.dataset)))[mask]
                ids = np.random.choice(optional_indexs, replace=False, size=100).tolist() # 选100个句子作为该batch候选

                batch_longest = 0
                batch_nexamps = 0 
                batch_ntokens = 0
                batch_ids = []

                def discharge():
                    nonlocal batch_longest
                    nonlocal batch_nexamps
                    nonlocal batch_ntokens
                    ret = batch_ids.copy()
                    batch_longest *= 0
                    batch_nexamps *= 0
                    batch_ntokens *= 0
                    batch_ids[:] = []
                    return ret
                while batch_ntokens < self.batch_size and len(ids) > 0:
                    idx = ids.pop()
                    size = self.dataset.size(self.dataset[idx])
                    cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
                    if cand_batch_ntokens > self.batch_size and batch_ids:
                        break
                    batch_longest = max(batch_longest, size)
                    batch_nexamps += 1
                    batch_ntokens = batch_longest * batch_nexamps
                    batch_ids.append(idx)
                yield discharge()