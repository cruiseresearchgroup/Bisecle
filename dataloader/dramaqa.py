import torch
from .base_dataset import BaseDataset
import json
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class DramaQA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train', type = None):
        super().__init__(args, tokenizer, split)
        
        data_list = json.load(open(f'/data/dramaqa/split_data/AnotherMissOhQA_{split}_set_{type}.json', "r"))
        self.data = pd.DataFrame(data_list)
        
        if 'vid' in self.data.columns:
            self.data.rename(columns={'vid': 'video'}, inplace=True)
        if 'correct_idx' in self.data.columns:
            self.data.rename(columns={'correct_idx': 'answer'}, inplace=True)
        
        self.data['type'] = type
        
        self.features = torch.load(f'data/dramaqa/clipvitl14.pth', weights_only=True)
        
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)', 4: '(E)'}
        self.num_options = 5
        self.qtype_mapping = {'TW': 0, 'DO': 1, 'DL': 2, 'CH': 3, 'CW': 4}
        
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
        weights = {k: 1.0 / v for k, v in type_counts.items()}
        return {self.qtype_mapping[k]: v * len(type_counts) for k, v in weights.items()}
        
    def _get_text(self, idx):
        sample = self.data.iloc[idx]
        question = sample["que"].capitalize().strip()
        if question[-1] != "?":
            question = question + "?"
        
        options = sample['answers']
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text
    
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

    def _get_video(self, video_id , idx):
        
        scene = True
        # Scene
        if video_id[-4:] == '0000':
            shots = self.data.iloc[idx]['shot_contained']
            start, end = shots[0], shots[1]

            for i in range(start, end+1):
                v_name = video_id[:-4] + f'{i:04}'

                if v_name not in self.features.keys(): 
                    print(v_name, " Not in features")
                    nxt_vid = torch.zeros(1, self.features_dim)
                else: nxt_vid = self.features[v_name].float()

                if i == start: video = nxt_vid
                else: video = torch.concat((video, nxt_vid), dim = 0)
        # Shot
        else:
            scene = False
            if video_id not in self.features.keys():
                print(video_id, "Not in freatures")
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
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
        else:
            video_len = self.max_feats

        return video, video_len, scene

    def __getitem__(self, idx):
        vid = self.data['video'].values[idx]
        qtype = self.qtype_mapping[self.data['type'].values[idx]]
        neg_qtypes = self._get_negative_qtypes(qtype)

        answer = self.data['answer'].values[idx]
        text = self._get_text(idx)
        
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        video, video_len, scene = self._get_video(f'{vid}', idx)
        
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
