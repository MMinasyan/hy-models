import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer
from torchdata.dataloader2 import DataLoader2
# from torch.utils.data import DataLoader
import torchdata.datapipes as dp
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import json
import itertools
import subprocess
import time
import torch._dynamo
torch._dynamo.config.verbose = True
# from transformers import Trainer, TrainingArguments, TrainerCallback
# import transformers
from modeling2 import HyCorrConfig, HyCorr
from utils import CharTokenizer
from preproc import clean_bytes
from training import compute_metrics, create_log_dict, print_log_dict
import torch.fx
from monitor import get_layers_data

def setup(rank, world_size):
    print("Setting up the distributed environment.")
    os.environ['MASTER_ADDR'] = 'localhost'  # Use the actual IP if across multiple machines
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Distributed environment setup complete.")


tokenizer = AutoTokenizer.from_pretrained("tokenizer")
VOCAB_SIZE = tokenizer.vocab_size

bytetokenizer = CharTokenizer(pad_token=0, unk_token=1).from_file('ByteToken')
char_vocab_size = bytetokenizer.get_vocab_size()

# MAX_LENGTH_W = 22
MAX_LENGTH_E = 512
MAX_LENGTH = 768
MAX_LENGTH_W = 22

NUM_HEAD = 16
NUM_LAYERS = 12
EMBED_DIM = NUM_HEAD * 64
FF_DIM = EMBED_DIM * 4
DROPOUT = 0.1

BATCH_SIZE = 14
N_GRAD = 4

LR = 0.0001
# LR = LR/N_GRAD


def split_text(text):
    text = text.split('[SEP]')
    assert len(text)==2
    return text[0], text[1]

def preprocess_t2t(batch, device=None):
    source_texts, target_texts = zip(*batch)

    source_texts = target_texts # for pre-training only
    source_input_ids = bytetokenizer.encode_2d_batch(source_texts, max_len=MAX_LENGTH_E, max_len_w=MAX_LENGTH_W)
    source_input_ids = torch.tensor(source_input_ids, dtype=torch.long, device=device)
    source_attention_masks = (source_input_ids.sum(dim=-1) != bytetokenizer.char2id['[PAD]']).long().to(device)

    target_texts = [clean_bytes(line) for line in target_texts]
    target_encodings = tokenizer(target_texts, add_special_tokens=True, padding='max_length', return_tensors='pt', max_length=MAX_LENGTH+1, truncation=True)
    target_input_ids = target_encodings['input_ids'].to(device)
    target_attention_masks = target_encodings['attention_mask'].to(device)

    labels = target_input_ids[:,1:].clone().to(device)
    labels[labels == tokenizer.pad_token_id] = -100

    target_input_ids = target_input_ids[:,:MAX_LENGTH]
    target_attention_masks = target_attention_masks[:,:MAX_LENGTH]

    return source_input_ids, source_attention_masks, target_input_ids, target_attention_masks, labels


def create_datapipe(file_path, batch_size=None, buffer_size=10000, device=None):
    datapipe = dp.iter.FileOpener(file_path, mode='rt')
    datapipe = dp.iter.LineReader(datapipe, return_path=False)
    # datapipe = dp.iter.Shuffler(datapipe, buffer_size=buffer_size)
    datapipe = datapipe.map(split_text)
    datapipe = datapipe.filter(lambda line: line[0].strip() != "" and line[1].strip() != "")
    datapipe = datapipe.batch(batch_size=batch_size, drop_last=True)
    datapipe = datapipe.map(lambda batch: preprocess_t2t(batch, device=device))
    return datapipe


train_dir = f'data/768t/train/'
train_files = [train_dir+fname for fname in os.listdir(train_dir)]
train_files = itertools.cycle(train_files)
print(train_files)

val_dir = f'data/768t/val/'
val_files = [val_dir+fname for fname in os.listdir(val_dir)]
val_files = itertools.cycle(val_files)
print(val_files)


def get_n_lines(filepath):
    result = subprocess.run(['wc', '-l', filepath], stdout=subprocess.PIPE, text=True)
    return int(result.stdout.split()[0])


def adjust_learning_rate(optimizer, epoch, step, warmup_steps):
    lr = LR * (0.5 ** epoch)
    if step < warmup_steps and epoch < 1:  # Warmup
        lr = lr * (step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def tile_inputs(source_input_ids, source_attention_masks, target_input_ids, target_attention_masks, labels):
    source_input_ids = torch.tile(source_input_ids, (2, 1, 1))
    source_attention_masks = torch.tile(source_attention_masks, (2, 1))
    target_input_ids = torch.tile(target_input_ids, (2, 1))
    target_attention_masks = torch.tile(target_attention_masks, (2, 1))
    labels = torch.tile(labels, (2, 1))
    return source_input_ids, source_attention_masks, target_input_ids, target_attention_masks, labels


def main():
    world_size = 2
    rank = int(os.getenv('LOCAL_RANK', '0'))
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    print(f"Rank {rank} entering training loop...")

    config=HyCorrConfig(
        hidden_dim=EMBED_DIM, 
        num_heads=NUM_HEAD, 
        num_layers=NUM_LAYERS,
        char_vocab_size=char_vocab_size,
        vocab_size=VOCAB_SIZE, 
        dropout=0.1, 
        encoder_length=MAX_LENGTH_E, 
        decoder_length=MAX_LENGTH,
        use_swiglu=True,
        char_embed_dim=128,
        char_num_filters=512,
    )

    model = HyCorr(config).to(rank)
    model.encoder_embedding.load_state_dict(torch.load('models/CE1024', weights_only=True))
    for param in model.encoder_embedding.parameters():
        param.requires_grad = False
    # model.load_state_dict(torch.load('models/'+model_name))
    # torch.cuda.empty_cache()
    # model.compile()
    ddp_model = DDP(model, device_ids=[device.index], gradient_as_bucket_view=True)
    print("Model setup complete.")
    model_name = f'ED_500w_768t_500m_rope_swg_b{BATCH_SIZE}'
    
    for name, child in ddp_model.named_children():
        num_params = sum(p.numel() for p in child.parameters())
        print(f"{name}: {num_params:,}")
    total_params = sum(p.numel() for p in ddp_model.parameters())
    print(f"{model_name}\nTotal: {total_params:,}")

    N_EPOCH = 2
    display_steps = 5
    log_steps = 1000

    log_path = f'logs/{model_name}_1'
    chpt_path = f'models/{model_name}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    optimizer = AdamW(ddp_model.parameters(), lr=LR)
    scaler = GradScaler()

    best_val_loss = float('inf')

    if rank == 0:
        layers_histogram = get_layers_data(ddp_model.module, epoch=1, step=0)
        with open(log_path+'/layers_hist.jsonl', "a") as f:
            f.write(json.dumps(layers_histogram) + "\n")
    for epoch in range(N_EPOCH):
        train_file = next(train_files)
        train_dataset = create_datapipe([train_file], batch_size=BATCH_SIZE, device=device)
        train_dataloader = DataLoader2(train_dataset)
        train_samples = get_n_lines(train_file)
        train_batches = train_samples // BATCH_SIZE
        print("Data loader setup complete.")

        ddp_model.train()
        print(f'==== Epoch {epoch+1}: Train ({train_batches} steps) {train_file} ====')
        loss_values = []
        acc_values = []
        start_time = time.time()

        for step, inputs in enumerate(train_dataloader):
            source_input_ids, source_attention_masks, target_input_ids, target_attention_masks, labels = inputs
            batch_len = source_input_ids.shape[0]
            if batch_len == 1:
                source_input_ids, source_attention_masks, target_input_ids, target_attention_masks, labels = tile_inputs(source_input_ids, source_attention_masks, target_input_ids, target_attention_masks, labels)
                batch_len = source_input_ids.shape[0]
                # print(f'After tail shape: {source_input_ids.shape}')
            source_input_ids = source_input_ids[rank*(batch_len//2):(rank+1)*(batch_len//2)]
            source_attention_masks = source_attention_masks[rank*(batch_len//2):(rank+1)*(batch_len//2)]
            target_input_ids = target_input_ids[rank*(batch_len//2):(rank+1)*(batch_len//2)]
            target_attention_masks = target_attention_masks[rank*(batch_len//2):(rank+1)*(batch_len//2)]
            labels = labels[rank*(batch_len//2):(rank+1)*(batch_len//2)]

            with autocast('cuda'):
                outputs = ddp_model(
                    source_input_ids,
                    target_input_ids,
                    attention_mask=source_attention_masks,
                    decoder_attention_mask=target_attention_masks,
                    labels=labels
                )
                loss = outputs['loss'].mean()
                if torch.isnan(loss).any().item():
                    print('!!! Found nan in loss !!!')
                    continue
                loss = loss / N_GRAD

            scaler.scale(loss).backward()
            if (step + 1) % N_GRAD == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(ddp_model.parameters(), clip_value=1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            accuracy = compute_metrics(outputs['logits'].permute(0, 2, 1).contiguous(), labels)

            adjust_learning_rate(optimizer, epoch, step, warmup_steps=10000)

            loss_values.append(loss.item()*N_GRAD)
            acc_values.append(accuracy)
            
            if (step+1) % display_steps == 0:
                avg_time=(time.time()-start_time)/len(loss_values)
                avg_time = torch.tensor(avg_time, dtype=torch.float32, device=device)
                local_loss_mean = torch.tensor(sum(loss_values) / len(loss_values), device=device)
                local_acc_mean = torch.tensor(sum(acc_values) / len(acc_values), device=device)

                # Use all_reduce to sum these local means across all GPUs
                dist.all_reduce(local_loss_mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_acc_mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_time, op=dist.ReduceOp.SUM)

                if rank == 0:
                    # Calculate the global mean of means
                    log_dict = create_log_dict(
                        timestamp=time.time(),
                        epoch=epoch+1,
                        step=step+1,
                        loss=local_loss_mean.item() / dist.get_world_size(),
                        accuracy=local_acc_mean.item() / dist.get_world_size(),
                        avg_time=avg_time.item() / dist.get_world_size(),
                        lr=optimizer.param_groups[0]['lr']
                        )
                    print_log_dict(log_dict)
                if (step+1) % log_steps == 0:
                    if rank==0:
                        with open(log_path+'/train.jsonl', 'a') as file:
                            json_string = json.dumps(log_dict)
                            file.write(json_string + '\n')

                        layers_histogram = get_layers_data(ddp_model.module, epoch+1, step+1)
                        with open(log_path+'/layers_hist.jsonl', "a") as f:
                            f.write(json.dumps(layers_histogram) + "\n")
                    
                    loss_values = []
                    acc_values = []
                    start_time = time.time()

        loss_values = []
        acc_values = []
        start_time = time.time()
        print('\n')

        val_file = next(val_files)
        val_dataset = create_datapipe([val_file], batch_size=BATCH_SIZE, device=device)
        val_dataloader = DataLoader2(val_dataset)
        val_samples = get_n_lines(val_file)
        val_batches = val_samples // BATCH_SIZE
        
        ddp_model.eval()
        with torch.no_grad(), autocast('cuda'):
            print(f'==== Epoch {epoch+1}: Validation ({val_batches} steps) ====')
            loss_values = []
            acc_values = []
            start_time = time.time()
            for val_step, inputs in enumerate(val_dataloader):
                source_input_ids, source_attention_masks, target_input_ids, target_attention_masks, labels = inputs
                batch_len = source_input_ids.shape[0]
                if batch_len == 1:
                    source_input_ids, source_attention_masks, target_input_ids, target_attention_masks, labels = tile_inputs(source_input_ids, source_attention_masks, target_input_ids, target_attention_masks, labels)
                    # print(f'After tail shape: {source_input_ids.shape}')
                source_input_ids = source_input_ids[rank*(batch_len//2):(rank+1)*(batch_len//2)]
                source_attention_masks = source_attention_masks[rank*(batch_len//2):(rank+1)*(batch_len//2)]
                target_input_ids = target_input_ids[rank*(batch_len//2):(rank+1)*(batch_len//2)]
                target_attention_masks = target_attention_masks[rank*(batch_len//2):(rank+1)*(batch_len//2)]
                labels = labels[rank*(batch_len//2):(rank+1)*(batch_len//2)]
                
                val_outputs = ddp_model(
                    source_input_ids,
                    target_input_ids,
                    attention_mask=source_attention_masks,
                    decoder_attention_mask=target_attention_masks,
                    labels=labels
                )
                val_loss = val_outputs['loss']
                val_accuracy = compute_metrics(val_outputs['logits'].permute(0, 2, 1).contiguous(), labels)
                loss_values.append(val_loss.item())
                acc_values.append(val_accuracy)

                if (val_step+1) % display_steps == 0:
                    avg_time=(time.time()-start_time)/len(loss_values)
                    avg_time = torch.tensor(avg_time, dtype=torch.float32, device=device)
                    local_loss_mean = torch.tensor(sum(loss_values) / len(loss_values), device=device)
                    local_acc_mean = torch.tensor(sum(acc_values) / len(acc_values), device=device)

                    # Use all_reduce to sum these local means across all GPUs
                    dist.all_reduce(local_loss_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(local_acc_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_time, op=dist.ReduceOp.SUM)

                    if rank == 0:
                        # Calculate the global mean of means
                        log_dict = create_log_dict(
                            timestamp=time.time(),
                            epoch=epoch+1,
                            step=val_step+1,
                            loss=local_loss_mean.item() / dist.get_world_size(),
                            accuracy=local_acc_mean.item() / dist.get_world_size(),
                            avg_time=avg_time.item() / dist.get_world_size()
                            )
                        print_log_dict(log_dict)

        if rank == 0:
            with open(log_path+'/val.jsonl', 'a') as file:
                json_string = json.dumps(log_dict)
                file.write(json_string + '\n')
        last_val_loss = sum(loss_values)/len(loss_values)
        loss_values = []
        acc_values = []
        start_time = time.time()
        

        last_val_loss_tensor = torch.tensor(last_val_loss, device=device)
        dist.all_reduce(last_val_loss_tensor, op=dist.ReduceOp.SUM)
        last_val_loss = last_val_loss_tensor.item() / dist.get_world_size()

        if rank == 0:
            if last_val_loss < best_val_loss:
                if 'models' not in os.listdir():
                    os.mkdir('models')
                torch.save(ddp_model.module.state_dict(), chpt_path)
                print(f'\nValidation loss {round(last_val_loss, 6)} < {round(best_val_loss, 6)}, saving state_dict to "{chpt_path}"')
                best_val_loss = last_val_loss
            else:
                print(f'\nValidation loss did not improve: {round(last_val_loss, 6)} >= {round(best_val_loss, 6)}')
        loss_values = []
        acc_values = []
        print('\n')
    cleanup()
    
def cleanup():
    dist.destroy_process_group()

import signal

# Function to handle SIGINT
def signal_handler(signal, frame):
    print("SIGINT received. Ignoring.")


if __name__ == '__main__':
    print("Starting script...")
    signal.signal(signal.SIGINT, signal_handler)
    main()

## Run with command:
## torchrun --nproc_per_node=2 ddp2.py