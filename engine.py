import torch
import math
import sys
import os
import json
from typing import Iterable
import util.misc as misc
import util.lr_sched as lr_sched
from torch.amp import autocast
import os
import json
import time
import numpy as np


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, args=None, current_qtype = None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = int(len(data_loader) / 4)
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    embedding_collection = []

    total_batches = len(data_loader)
    save_batch_indices = set([int(total_batches * i / 5) for i in range(5)])
    
    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)



        save_embeddings = data_iter_step in save_batch_indices
        
        if save_embeddings and 'qtype' in data and current_qtype is not None:
            # Forward pass with embedding saving
            losses, embedding_dict = model(data, save_embeddings=True, 
                                          current_epoch=epoch, 
                                          question_type=current_qtype)
            embedding_collection.append(embedding_dict)
        else:
            # Normal forward pass
            losses = model(data)
        
        # compute loss
        with autocast(device_type="cuda"):
            loss = losses['vqa_loss'] 
            if args.vaq:
                loss = loss + losses['vaq_loss']
            if args.qav:
                loss = loss + losses['qav_loss']
            
            if 'tqcp_loss' in losses:
                loss = loss + losses['tqcp_loss']
                tqcp_loss_value = losses['tqcp_loss'].item()
                metric_logger.update(tqcp_loss=tqcp_loss_value)

        loss_value = loss.item()
        vqa_loss_value = losses['vqa_loss'].item()
        
        if args.vaq:
            vaq_loss_value = losses['vaq_loss'].item()
            metric_logger.update(vaq_loss=vaq_loss_value)
        if args.qav:
            qav_loss_value = losses['qav_loss'].item()
            metric_logger.update(qav_loss=qav_loss_value)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(vqa_loss=vqa_loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def val_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, args=None, task_type=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = max(1, int(len(data_loader) / 4))

    wrong_list = []

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        answer = data['answer'].cuda()
        bsz = answer.shape[0]

        with torch.no_grad():
            with autocast(device_type="cuda"):
                logits = model(data, inference=True)
        
        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)
        eval_flag = (answer == prediction)
        acc = eval_flag.sum().item() / bsz
        
        misc.log_qtype(data, eval_flag, metric_logger, args)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(n=bsz, acc=acc)
        
        eval_list = eval_flag.cpu().tolist()
        pred_list = prediction.cpu().tolist()
        answer_list = answer.cpu().tolist()
        
        for i in range(bsz):
            if not eval_list[i]:
                if args.dataset == 'nextqa':
                    wrong_list.append({
                        "video": str(data['vid'][i]) if isinstance(data['vid'][i], (np.int64, int)) else data['vid'][i],
                        "prediction": int(pred_list[i]),
                        "ground_truth": int(answer_list[i])
                    })
                else:
                    wrong_list.append({
                        "question_id": data['question_id'][i][0],
                        "video_id": data['vid'][i],
                        "prediction": pred_list[i],
                        "ground_truth": answer_list[i]
                    })
        

        

    import time
    readable_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    log_dir = "/nextqa_example_log/"
    os.makedirs(log_dir, exist_ok=True)
    task_str = f"_{task_type}" if task_type is not None else ""
    file_name = os.path.join(log_dir, f"error_log_epoch{epoch}_{task_type}_{readable_time}.json")
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(wrong_list, f, indent=4, ensure_ascii=False)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
