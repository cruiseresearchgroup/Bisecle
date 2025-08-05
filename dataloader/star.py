import torch
from .base_dataset import BaseDataset
import json
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class STAR(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train', type = None):
        super().__init__(args, tokenizer, split)
        data_list = json.load(open(f'./data/star/split_files/Star_{split}_{type}.json', 'r'))
        self.data = pd.DataFrame(data_list)
        
        
        self.features = torch.load(f'./data/star/clipvitl14.pth', weights_only=True )
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.qtype_mapping = {'Interaction': 0, 'Sequence': 1, 'Prediction': 2, 'Feasibility': 3}
        self.num_options = 4
        print(f"Num {split} data: {len(self.data)}") 
        
        self.type_weights = self._calculate_type_weights()
        self.cached_features = self._preload_features()
        self.negative_sample_size = args.negative_sample_size


    def _get_text(self, idx):
        row = self.data.iloc[idx]
        question = row["question"].capitalize().strip()
        if question[-1] != "?":
            question = question + "?"
            
        options = {x['choice_id']: x['choice'] for x in row['choices']}
        options = [options[i] for i in range(self.num_options)]
        answer = options.index(row['answer'])
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text, answer


    def _get_video(self, video_id, start, end):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id][start: end +1, :].float()
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

        return video, video_len

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vid = row['video_id']
        question_id = row['question_id']
        qtype = self.qtype_mapping[row['question_id'].split('_')[0]]
        neg_qtypes = self._get_negative_qtypes(qtype)
        
        text, answer = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        start, end = round(row['start']), round(row['end'])
        video, video_len = self._get_video(f'{vid}', start, end)
        type_weight = self.type_weights[qtype]
        return {"vid": vid, 
                "question_id": [question_id],
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
                "type_weight": type_weight}



    def __len__(self):
        return len(self.data)
    
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
        question_types = self.data['question_id'].apply(lambda x: x.split('_')[0])
        type_counts = question_types.value_counts()
        weights = {k: 1.0 / v for k, v in type_counts.items()}
        return {self.qtype_mapping[k]: v * len(type_counts) for k, v in weights.items()}

    
    def _preload_features(self):
        cached = {}
        for vid in self.data['video_id'].unique():
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

    
