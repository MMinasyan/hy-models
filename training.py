import os
import torch
from torch.cuda.amp import GradScaler, autocast
import json


def compute_metrics(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    mask = (labels != -100)

    correct_per_example = ((predictions == labels) & mask).float()
    total_per_example = mask.float()
    example_accuracies = correct_per_example.sum() / total_per_example.sum()
    return example_accuracies.item()

def create_log_dict(timestamp, epoch, step, loss, accuracy, avg_time=None, lr=None):
    log_dict = {
        'Timestamp': timestamp,
        'Epoch': epoch,
        'Step': step,
        'Loss': loss,
        'Accuracy': accuracy,
    }
    if not avg_time is None:
        log_dict['avg_time'] = avg_time
    if not lr is None:
        log_dict['lr'] = lr
    return log_dict

def print_log_dict(logs, end='\r'):
    messages = []
    for k in logs.keys():
        if k == 'avg_time':
            messages.append(f'{k}: {round(logs[k], 3)}')
        elif k == 'lr':
            messages.append(f'{k}: {round(logs[k], 6)}')
        elif k not in ['Timestamp', 'Epoch']:
            messages.append(f'{k}: {round(logs[k], 4)}')
    print('--- ' + ', '.join(messages) + ' ---', end=end)


def train(model, n_epochs, n_grad, optimizer, dataloader_train, train_steps, lr_function, dataloader_val=None, val_steps=None, start_epoch=0, display_steps=1, log_path=None, log_steps=1000, chpt_path=None):
    if log_path:
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    scaler = GradScaler()

    last_epoch_val_loss = float('inf')

    for epoch in range(start_epoch, n_epochs):
        if callable(dataloader_train):
            train_dataloader = dataloader_train()
        else:
            train_dataloader = dataloader_train

        model.train()
        print(f'==== Epoch {epoch+1}: Train ({train_steps} steps) ====')
        loss_values = []
        acc_values = []
        for step, inputs in enumerate(train_dataloader):
            input_ids, attention_mask, labels = inputs
            
            # Mixed Precision
            with autocast():
                outputs = model(
                    input_ids,
                    attention_mask,
                    labels,
                )
                loss = outputs['loss']

            scaler.scale(loss).backward()
            if (step + 1) % n_grad == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            accuracy = compute_metrics(outputs['logits'].permute(0, 2, 1).contiguous(), labels)

            lr_function(optimizer, epoch, step)

            loss_values.append(loss.item())
            acc_values.append(accuracy)
            
            if (step+1) % display_steps == 0:
                log_dict = create_log_dict(
                    epoch+1,
                    step+1,
                    sum(loss_values)/len(loss_values),
                    sum(acc_values)/len(acc_values),
                    optimizer.param_groups[0]['lr']
                    )
                print_log_dict(log_dict)
                if (step+1) % (log_steps) == 0:
                    if log_path:
                        with open(log_path+'/train.jsonl', 'a') as file:
                            json_string = json.dumps(log_dict)
                            file.write(json_string + '\n')
                    loss_values = []
                    acc_values = []
        loss_values = []
        acc_values = []
        print('\n')

        if dataloader_val:
            if callable(dataloader_val):
                val_dataloader = dataloader_val()
            else:
                val_dataloader = dataloader_val
            model.eval()
            with torch.no_grad(), autocast():
                print(f'==== Epoch {epoch+1}: Validation ({val_steps} steps) ====')
                loss_values = []
                acc_values = []
                for val_step, inputs in enumerate(val_dataloader):
                    input_ids, attention_mask, labels = inputs
                    val_outputs = model(
                        input_ids,
                        attention_mask,
                        labels,
                    )
                    val_loss = val_outputs['loss']
                    val_accuracy = compute_metrics(val_outputs['logits'].permute(0, 2, 1).contiguous(), labels)
                    loss_values.append(val_loss.item())
                    acc_values.append(val_accuracy)
                    if (val_step+1) % display_steps == 0:
                        log_dict = create_log_dict(
                            epoch+1,
                            val_step+1,
                            sum(loss_values)/len(loss_values),
                            sum(acc_values)/len(acc_values),
                            optimizer.param_groups[0]['lr']
                            )
                        print_log_dict(log_dict)
                        if (val_step+1) % (log_steps) == 0:
                            if log_path:
                                with open(log_path+'/val.jsonl', 'a') as file:
                                    json_string = json.dumps(log_dict)
                                    file.write(json_string + '\n')
                            last_val_loss = sum(loss_values)/len(loss_values)
                            loss_values = []
                            acc_values = []
                        print_log_dict(log_dict)
            if chpt_path:
                if last_val_loss < last_epoch_val_loss:
                    torch.save(model.state_dict(), chpt_path)
                    print(f'\nValidation loss {round(last_val_loss, 6)} < {round(last_epoch_val_loss, 6)}, saving state_dict to "{chpt_path}"')

            last_epoch_val_loss = last_val_loss
            loss_values = []
            acc_values = []
            print('\n')
        else:
            if chpt_path:
                torch.save(model.state_dict(), chpt_path)
                print(f'\nsaving state_dict to "{chpt_path}"')

        