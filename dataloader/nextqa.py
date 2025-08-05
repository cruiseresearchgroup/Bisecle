import torch
from .base_dataset import BaseDataset
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class NextQA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train', type=None):
        super().__init__(args, tokenizer, split)
        self.data = pd.read_csv(f'/data/nextqa/split_data/{split}_{type}.csv')
        self.features = torch.load(f'/data/{args.dataset}/clipvitl14.pth', weights_only= True)
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        self.qtype_mapping = {'TP': 0, 'CW': 1, 'DC': 2, 'TC': 3, 'DL': 4, 'DO': 5, 'TN': 6, 'CH': 7}
        
        self.type_weights = self._calculate_type_weights()
        self.cached_features = self._preload_features()
        self.negative_sample_size = args.negative_sample_size
        
        print(f"Num {split} data: {len(self.data)}")
        
    def _get_negative_qtypes(self, current_qtype):
        if hasattr(self, 'processed_qtypes') and self.processed_qtypes and len(self.processed_qtypes) > 1:
            negative_candidates = [self.qtype_mapping[q] for q in self.processed_qtypes
                                if self.qtype_mapping[q] != current_qtype]
        else:
            negative_candidates = []
        
        if len(negative_candidates) > self.negative_sample_size:
            negative_candidates = np.random.choice(negative_candidates, self.negative_sample_size, replace=False)
        return torch.tensor(negative_candidates, dtype=torch.long)

    def _calculate_type_weights(self):
        type_counts = self.data['type'].value_counts()
        weights = dict(1.0 / type_counts)
        return {self.qtype_mapping[k]: v * len(type_counts) for k, v in weights.items()}
    
    def _preload_features(self):
        cached = {}
        for vid in self.data['video'].unique():
            if vid in self.features:
                video = self.features[vid].float()
                if len(video) > self.max_feats:
                    indices = torch.linspace(0, len(video)-1, self.max_feats).long()
                    video = video[indices]
                elif len(video) < self.max_feats:
                    video = torch.cat([
                        video,
                        torch.zeros(self.max_feats - len(video), self.features_dim)
                    ], dim=0)
                cached[vid] = video
        return cached
        
    def _get_text(self, idx):
        question = self.data["question"].values[idx].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = [self.data[f'a{i}'].values[idx] for i in range(self.num_options)]
        qtype = self.data['type'].values[idx]
        q_text = f"Question Type: {qtype}\n Question: {question}\n"
        o_text = "Choices: \n"
        
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text

    def _get_video(self, video_id):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id].float()
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], dim=0)
        else:
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        vid = self.data['video'].values[idx]
        qtype = self.qtype_mapping[self.data['type'].values[idx]]
        neg_qtypes = self._get_negative_qtypes(qtype)

        answer = self.data['answer'].values[idx]
        text = self._get_text(idx)
        
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        video, video_len = self._get_video(f'{vid}')
        
        type_weight = self.type_weights[qtype]
        
        return {
            "vid": vid,
            "video": video,
            "video_len": video_len,
            "text": text,
            "text_id": text_id,
            "label": label,
            "video_start": video_start,
            "video_index": video_index,
            "label_mask": label_mask,
            "qid": idx,
            "answer": answer,
            "qtype": qtype,
            "neg_qtypes": neg_qtypes,
            "type_weight": type_weight
        }

    def __len__(self):
        return len(self.data)
