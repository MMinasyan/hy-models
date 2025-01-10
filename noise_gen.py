import numpy as np
import copy
import torch
from torch import nn
from utils import CharTokenizer
from transformers import AutoTokenizer
# import tensorflow as tf
import math
import re
import random
import time
import preproc
from ortho_patterns import patterns_dicts
import socket
import pickle
from translate_simple import build_nllb, back_translate_texts


char_probas = np.load('noise_gen/char_probas.npy')

def modify_char_probas(char_probas_input):
    char_probas = char_probas_input.copy()
    n = char_probas.shape[0]

    cumulative_sums = np.cumsum(char_probas, axis=1)
    random_thresholds = np.random.rand(char_probas.shape[0], 1)
    selected_indices = (cumulative_sums >= random_thresholds).argmax(axis=1)

    new_values = np.random.rand(n)
    char_probas[np.arange(n), selected_indices] = new_values

    sum_excluding_new = np.sum(char_probas, axis=1) - char_probas[np.arange(n), selected_indices]

    scaling_factors = (1 - new_values) / sum_excluding_new

    char_probas *= scaling_factors[:, np.newaxis]
    char_probas[np.arange(n), selected_indices] = new_values
    return char_probas

current_char_probas = modify_char_probas(char_probas)

tokenizer = AutoTokenizer.from_pretrained("wp_tokenizer")
bytetokenizer = CharTokenizer(pad_token=0, unk_token=1).from_file('ByteToken')


class CharModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_length, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_length + 1, hidden_dim)
        self.max_length = max_length

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=nn.GELU(),
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        position_indices = torch.arange(1, max_length + 1).expand(1, -1)
        self.register_buffer('position_indices', position_indices)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask=None, labels=None):
        if attention_mask is not None:
            attention_mask = attention_mask == 0
        
        encoder_embeddings = self.embedding(input_ids)
        position_embeddings = self.pos_embedding(self.position_indices)
        encoder_embeddings = encoder_embeddings + position_embeddings[:,:input_ids.size(1)]

        encoder_outputs = self.encoder(encoder_embeddings, src_key_padding_mask=attention_mask)

        output_logits = self.output_layer(encoder_outputs)
        output_logits = output_logits.permute(0, 2, 1)
        
        outputs = {'logits': output_logits}
        if labels is not None:
            loss = self.loss_fct(output_logits, labels)
            outputs['loss'] = loss
        
        return outputs

    def generate(self, input_ids, mask_start, mask_end, exclude_tokens=None, temperature=0.9):
        with torch.no_grad():
            first_iteration = False
            if exclude_tokens is not None:
                first_iteration = True

            current_pos = mask_start.clone()
            finished = current_pos >= mask_end
            
            while not finished.all():
                logits = self(input_ids)['logits'] # (batch_size, vocab_size, max_length)
                batch_size, vocab_size, max_length = logits.shape
                
                logits_scaled = logits / temperature
                logits_scaled = logits_scaled.permute(0, 2, 1) # (batch_size, max_length, vocab_size)
                
                probs = torch.softmax(logits_scaled, dim=2)
                probs = probs + 1e-10
                probs[:,:,:2] = 0. # PAD and UNK
                probs[:,:,-1] = 0. # MASK
                
                if first_iteration:
                    batch_indices = torch.arange(batch_size)
                    probs[batch_indices, :, exclude_tokens] = 0.
                
                probs = probs / probs.sum(dim=2, keepdim=True)

                probs = probs.reshape(-1, vocab_size)
                preds = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)
                preds = preds.reshape(batch_size, max_length)

                active = current_pos < mask_end
                batch_indices = torch.nonzero(active).squeeze()
                input_ids[batch_indices, current_pos[batch_indices]] = preds[batch_indices, current_pos[batch_indices]]

                current_pos[batch_indices] += 1
                finished = current_pos >= mask_end
                first_iteration = False


class TokenModel(CharModel):
    def generate(self, input_ids, mask_start, mask_end, exclude_tokens=None, temperature=0.9):
        with torch.no_grad():
            first_iteration = False
            if exclude_tokens is not None:
                first_iteration = True

            current_pos = mask_start.clone()
            finished = current_pos >= mask_end
            
            while not finished.all():
                logits = self(input_ids)['logits'] # (batch_size, vocab_size, max_length)
                batch_size, vocab_size, max_length = logits.shape
                
                logits_scaled = logits / temperature
                logits_scaled = logits_scaled.permute(0, 2, 1) # (batch_size, max_length, vocab_size)
                
                probs = torch.softmax(logits_scaled, dim=2)
                probs = probs + 1e-10
                probs[:, :, tokenizer.all_special_ids] = 0.
                
                if first_iteration:
                    batch_indices = torch.arange(batch_size)
                    probs[batch_indices, :, exclude_tokens] = 0.
                
                probs = probs / probs.sum(dim=2, keepdim=True)

                probs = probs.reshape(-1, vocab_size)
                preds = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)
                preds = preds.reshape(batch_size, max_length)

                active = current_pos < mask_end
                batch_indices = torch.nonzero(active).squeeze()
                input_ids[batch_indices, current_pos[batch_indices]] = preds[batch_indices, current_pos[batch_indices]]

                current_pos[batch_indices] += 1
                finished = current_pos >= mask_end
                first_iteration = False


def build_char_gen(model_name, max_length=48, trained=False, eval_mode=True, verbose=False, device=None):
    char_model = CharModel(
        vocab_size=bytetokenizer.get_vocab_size()+1,
        hidden_dim=64*6,
        num_heads=6,
        num_layers=6,
        max_length=max_length,
        dropout=0.1
        ).to(device)
    if trained:
        char_model.load_state_dict(torch.load('ckpts/'+model_name))

    if verbose:
        for name, child in char_model.named_children():
            num_params = sum(p.numel() for p in child.parameters())
            print(f"{name}: {num_params:,}")
    
    total_params = sum(p.numel() for p in char_model.parameters())
    print(f"\"{model_name}\" - N_param: {total_params:,}")

    if eval_mode:
        char_model.eval()

    return char_model

def build_token_gen(model_name, max_length=48, trained=False, eval_mode=True, verbose=False, device=None):
    token_model = TokenModel(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=64*8,
        num_heads=8,
        num_layers=8,
        max_length=max_length,
        dropout=0.1
        ).to(device)
    if trained:
        token_model.load_state_dict(torch.load('ckpts/'+model_name))

    if verbose:
        for name, child in token_model.named_children():
            num_params = sum(p.numel() for p in child.parameters())
            print(f"{name}: {num_params:,}")
    
    total_params = sum(p.numel() for p in token_model.parameters())
    print(f"\"{model_name}\" - N_param: {total_params:,}")

    if eval_mode:
        token_model.eval()

    return token_model


# def translate_and_back(batch, port=9999):
#     data_bytes = pickle.dumps(batch)

#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     host = socket.gethostname()

#     s.connect((host, port))

#     s.sendall(data_bytes)
#     s.shutdown(socket.SHUT_WR)

#     response_chunks = []
#     while True:
#         chunk = s.recv(16384)
#         if not chunk:
#             break
#         response_chunks.append(chunk)
#     response_bytes = b''.join(response_chunks)

#     response_received = pickle.loads(response_bytes)
#     s.close()
#     return response_received

n_values = np.arange(1, 31)
probas_long = 1 / n_values ** 1.8
probas_long[0] = 0.
probas_long = probas_long / np.sum(probas_long)

n_values = np.arange(1, 11)
probas_short = 1 / n_values ** 3
probas_short[0] = 0.
probas_short = probas_short / np.sum(probas_short)

probas_long_t = 1 / np.arange(1, 18) ** 2
probas_long_t[0] = 0.
probas_long_t = probas_long_t / np.sum(probas_long_t)

probas_short_t = 1 / math.e ** np.arange(1, 9)
probas_short_t[0] = 0.
probas_short_t = probas_short_t / np.sum(probas_short_t)


def get_n(probas):
    return np.random.choice(len(probas), p=probas)

def remove_chars(text, start_pos):
    n = get_n(probas_long)
    return text[:start_pos] + text[start_pos+n:], start_pos

def swap_chars(text, start_pos):
    n = get_n(probas_short)
    n = min(len(text)-start_pos-1, n)
    pos2 = start_pos + n
    return text[:start_pos] + text[pos2] + text[start_pos+1:pos2] + text[start_pos] + text[pos2+1:],  pos2 + 1

def insert_chars(text, start_pos):
    n = get_n(probas_long)
    placeholder = text[:start_pos+n][-n:]
    return text[:start_pos] + placeholder + text[start_pos:], start_pos + n

def replace_chars(text, start_pos):
    n_orig = get_n(probas_long)
    n = get_n(probas_long)
    placeholder = text[:start_pos+n_orig][-n:]
    return text[:start_pos] + placeholder + text[start_pos + n_orig:], start_pos + n

def replace_1char(text, pos, current_char_probas):
    orig_id = bytetokenizer.token_to_id(text[pos])
    if orig_id is None:
        return text
    if orig_id:
        new_id = np.random.choice(a=current_char_probas.shape[1], p=current_char_probas[orig_id,:])

    return text[:pos] + bytetokenizer.id_to_token(new_id) + text[pos+1:]

def get_char_prompt(chars, start_pos, end_pos, max_length=48):
    end_pos = min(end_pos, len(chars))
    assert end_pos - start_pos < max_length, f'[start_pos, end_pos): [{start_pos}, {end_pos}) has length more than max_length: {max_length}'
    max_length = min(len(chars), max_length)
    n_before = start_pos
    n_after = len(chars) - end_pos
    chars_to_add = max_length - (end_pos - start_pos)
    n_add_before = chars_to_add - min(n_after, 5)
    n_add_before = random.randint(min(5, n_add_before), n_add_before)
    n_add_before = min(n_before, n_add_before)
    n_add_after = min(n_after, chars_to_add - n_add_before)
    n_add_before += chars_to_add - (n_add_before + n_add_after)
    assert n_add_before+(end_pos - start_pos) <= max_length, f'generated mask end: {n_add_before+(end_pos - start_pos)} has length more than max_length: {max_length}\n\
        text length: {len(chars)}, start_pos, end_pos : {start_pos, end_pos},'
    chars_segment = chars[start_pos-n_add_before:end_pos+n_add_after]
    mask_start = n_add_before
    mask_end = n_add_before+(end_pos - start_pos)
    exclude_chars = chars_segment[mask_start]
    return chars_segment, mask_start, mask_end, start_pos, end_pos, exclude_chars

def preproc_char_seqments(segments, mask_start, mask_end, mask_id, max_length=48, exclude_tokens=None):
    input_ids = [bytetokenizer.encode_1d(text, max_len=max_length) for text in segments]
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    if exclude_tokens:
        exclude_tokens = bytetokenizer.encode_1d(exclude_tokens)
        exclude_tokens = torch.tensor(exclude_tokens, dtype=torch.long)

    mask_start = torch.tensor(mask_start, dtype=torch.long)
    mask_end = torch.tensor(mask_end, dtype=torch.long)

    seq_range = torch.arange(max_length).unsqueeze(0)
    mask = (seq_range >= mask_start.unsqueeze(1)) & (seq_range < mask_end.unsqueeze(1))
    input_ids[mask] = mask_id
    return input_ids, mask_start, mask_end, exclude_tokens

def remove_tokens(tokens, start_pos):
    n = get_n(probas_long_t)
    # print(f'Pos {start_pos}:{start_pos+n}, Remove {tokenizer.decode(tokens[start_pos:start_pos+n], skip_special_tokens=True)}')
    return tokens[:start_pos] + tokens[start_pos+n:], start_pos

def swap_tokens(tokens, start_pos):
    n = get_n(probas_short_t)
    n = min(len(tokens)-start_pos-1, n)
    pos2 = start_pos + n
    # print(f'Pos {start_pos} <-> {pos2}, Swap {tokenizer.convert_ids_to_tokens(tokens[start_pos])} <-> {tokenizer.convert_ids_to_tokens(tokens[pos2])}')
    return tokens[:start_pos] + [tokens[pos2]] + tokens[start_pos+1:pos2] + [tokens[start_pos]] + tokens[pos2+1:],  pos2 + 1

def insert_tokens(tokens, start_pos):
    n = get_n(probas_long_t)
    mask = [tokenizer.mask_token_id] * n
    return tokens[:start_pos] + mask + tokens[start_pos:], start_pos + n

def replace_tokens(tokens, start_pos):
    n_orig = get_n(probas_long_t)
    n = get_n(probas_long_t)
    mask = [tokenizer.mask_token_id] * n
    return tokens[:start_pos] + mask + tokens[start_pos + n_orig:], start_pos + n

# def remove_tokens(text, tokens, token_mapping, start_pos):
#     n = get_n(probas_long_t)
#     start_pos_c = token_mapping[start_pos][0]
#     end_pos_c = token_mapping[start_pos+n-1][1]
#     text = text[:start_pos_c] + text[end_pos_c:]
    
#     tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
#     tokens, token_mapping = tokens['input_ids'], tokens['offset_mapping']
#     return text, tokens, token_mapping, start_pos

# def swap_tokens(text, tokens, token_mapping, start_pos):
#     n = get_n(probas_short_t)
#     n = min(len(tokens)-start_pos-1, n)
#     pos2 = start_pos + n
#     start_pos_c = token_mapping[start_pos]
#     pos2_c = token_mapping[start_pos+n]
#     token_str1 = text[start_pos_c[0]:start_pos_c[1]]
#     token_str2 = text[pos2_c[0]:pos2_c[1]]
#     text = text[:start_pos_c[0]] + token_str2 + text[start_pos_c[1]:pos2_c[0]] + token_str1 + text[pos2_c[1]:]

#     tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
#     tokens, token_mapping = tokens['input_ids'], tokens['offset_mapping']
#     return text, tokens, token_mapping, pos2 + 1

# def insert_tokens(text, tokens, token_mapping, start_pos):
#     n = 2
#     mask_token = text[token_mapping[start_pos][0]:token_mapping[start_pos][1]]
#     if tokenizer.convert_ids_to_tokens(tokens[start_pos]).startswith('##'):
#         mask_token = ' ' + mask_token

#     pos_c = token_mapping[start_pos][0]
#     text[:pos_c] + mask_token + mask_token*(n-1) + text[pos_c:]

#     tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
#     tokens, token_mapping = tokens['input_ids'], tokens['offset_mapping']
#     return text, tokens, token_mapping, start_pos + n

# def replace_tokens(text, tokens, token_mapping, start_pos):
#     n_orig = get_n(probas_long_t)
#     n = get_n(probas_long_t)
#     pos_c = token_mapping[start_pos][0]
#     end_pos_c = token_mapping[start_pos + n_orig][0]
#     if n > n_orig:
#         mask = '[MASK]' * (n-n_orig)
#         text = text[:pos_c] + mask + text[pos_c:]
#     else:
#         n_diff = n_orig - n
#         end_pos_c = token_mapping[start_pos + n_diff][0]
#         if n_diff != 0 and tokenizer.convert_ids_to_tokens(tokens[start_pos + n_diff])[:2]!='##':
#             text = text[:pos_c] + ' ' + text[end_pos_c:]
#         else:
#             text = text[:pos_c] + text[end_pos_c:]

#     tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
#     tokens, token_mapping = tokens['input_ids'], tokens['offset_mapping']
#     return text, tokens, token_mapping, start_pos + n

def get_token_prompt(tokens, start_pos, end_pos, max_length=48):
    end_pos = min(end_pos, len(tokens))
    assert end_pos - start_pos < max_length, f'[start_pos, end_pos): [{start_pos}, {end_pos}) has length more than max_length: {max_length}'
    max_length = min(len(tokens), max_length)
    n_before = start_pos
    n_after = len(tokens) - end_pos
    tokens_to_add = max_length - (end_pos - start_pos)
    n_add_before = tokens_to_add - min(n_after, 5)
    n_add_before = random.randint(min(5, n_add_before), n_add_before)
    n_add_before = min(n_before, n_add_before)
    n_add_after = min(n_after, tokens_to_add - n_add_before)
    n_add_before += tokens_to_add - (n_add_before + n_add_after)
    assert n_add_before+(end_pos - start_pos) <= max_length, f'generated mask end: {n_add_before+(end_pos - start_pos)} has length more than max_length: {max_length}\n\
        text length: {len(tokens)}, start_pos, end_pos : {start_pos, end_pos},'
    tokens_segment = tokens[start_pos-n_add_before:end_pos+n_add_after]
    mask_start = n_add_before
    mask_end = n_add_before+(end_pos - start_pos)
    exclude_tokens = tokens_segment[mask_start]
    return tokens_segment, mask_start, mask_end, start_pos, end_pos, exclude_tokens

def preproc_token_seqments(segments, mask_start, mask_end, max_length=48, exclude_tokens=None):
    input_ids = [segment + [tokenizer.pad_token_id] * (max_length-len(segment)) for segment in segments]
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    if exclude_tokens:
        exclude_tokens = bytetokenizer.encode_1d(exclude_tokens)
        exclude_tokens = torch.tensor(exclude_tokens, dtype=torch.long)

    mask_start = torch.tensor(mask_start, dtype=torch.long)
    mask_end = torch.tensor(mask_end, dtype=torch.long)

    seq_range = torch.arange(max_length).unsqueeze(0)
    mask = (seq_range >= mask_start.unsqueeze(1)) & (seq_range < mask_end.unsqueeze(1))
    input_ids[mask] = tokenizer.mask_token_id
    return input_ids, mask_start, mask_end, exclude_tokens

def gen_ortho_mistakes(text, literacy_rate=1.):
    random.shuffle(patterns_dicts)
    # literacy_rate = random.random()
    mistake_rate = 1 #random.random()
    # print(literacy_rate)
    #mistake_dict = {}
    changes = []
    for pattern_dict in patterns_dicts:
        pattern = pattern_dict['pattern']
        #mistake_dict[pattern] = 0
        compiled_pattern = re.compile(pattern, re.IGNORECASE if pattern_dict['ignore_case'] else 0)
        # Preparing options and probabilities, including no replacement
        replacement_options = [replacement for replacement, _ in pattern_dict['replacement']]
        probabilities = [probability for _, probability in pattern_dict['replacement']]
        probabilities = [mistake_rate * p if random.random() < literacy_rate*p**(1/2) else 0. for p in probabilities]
        probabilities = [(min(p, 0.25) * 2.) ** ((random.random()**3)**(1.6)) for p in probabilities]
        # print(pattern, ':', probabilities)
        # print(replacement_options, probabilities)
        total_probability = sum(probabilities)
        # Add 'None' option
        replacement_options.append(None)
        probabilities.append(1 - total_probability)

        matches = sorted(compiled_pattern.finditer(text), key=lambda x: x.start(), reverse=True)
        for match in matches:
            start, end = match.start(), match.end()
            matched_text = text[start:end]
            # Select a replacement (or no replacement) based on probabilities
            selected_replacement, = random.choices(replacement_options, weights=probabilities, k=1)
            # Apply replacement
            if selected_replacement:
                #mistake_dict[pattern] += 1

                if callable(selected_replacement):  # If function
                    try:
                        replacement_text = selected_replacement(match.group())
                    except:
                        print(match.group(), selected_replacement)
                else:  # If string
                    replacement_text = selected_replacement
                changes.append([start, match.group(), replacement_text])
                # Replace text
                # print(f'"{text[start:end]}" -> "{replacement_text}" | | "{text[start-7:end+7]}" -> "{text[start-7:start]+replacement_text+text[end:end+7]}" | | "{pattern}" -> "{selected_replacement}"')
                text = text[:start] + replacement_text + text[end:]
    return text, len(changes)#, mistake_dict


def modify_texts(texts, char_model, token_model, translate_model, max_char_requests=64, max_token_requests=32, char_mod_rate=0.05, token_mod_rate=0.1, ortho_mod_rate=1.): 
    char_changes = 0
    ortho_changes = 0
    token_changes = 0
    # CHAR-LEVEL
    infer_times = 0
    char_infer_times = 0
    modification_requests = []
    token_requests = []
    max_token_len = token_model.max_length
    max_char_len = char_model.max_length

    start_time = time.time()
    texts = [text.replace('[NEWLINE]', '\n') for text in texts]
    # texts = translate_and_back(texts)
    texts = back_translate_texts(texts, model1=translate_model, batch_size=32, max_len=200, proba=0.1)
    translate_times = time.time() - start_time
    start_time = time.time()
    texts = tokenizer(texts, add_special_tokens=False)['input_ids']
    for tn in range(len(texts)):
        # TOKEN-LEVEL
        init_prob = token_mod_rate  ** (1. + random.random())
        # print(f'mod_rate={init_prob}')
        mod_probs = np.random.rand(4) * np.array([0.93, 0.01, 0.04, 0.02]) # replace, remove, insert, swap
        mod_probs = mod_probs/mod_probs.sum()
        pos = 0
        while pos < len(texts[tn]):
            # print(pos)
            if random.random() < init_prob:
                # print(random.random(), '<', init_prob)
                token_changes += 1
                next_token = texts[tn][pos]
                mod_func = random.choices(
                    [replace_tokens, remove_tokens, insert_tokens, swap_tokens],
                    weights=mod_probs, k=1
                )[0]
                
                texts[tn], new_pos = mod_func(texts[tn], pos)
                # print(pos, mod_func.__name__)

                if mod_func in [replace_tokens, insert_tokens]:
                    next_token = 0 if mod_func == insert_tokens else next_token
                    token_requests.append((tn, pos, new_pos, next_token))
                
                if len(token_requests) == max_token_requests:
                    # print('n_requests', len(token_requests))
                    prompting = [get_token_prompt(texts[t_n], start, end, max_length=max_token_len) for t_n, start, end, next_token in token_requests]
                    prompts, mask_start, mask_end, start_pos, end_pos, next_tokens= map(list, zip(*prompting))

                    input_ids, mask_start, mask_end, exclude_tokens = preproc_token_seqments(
                        prompts,
                        mask_start=mask_start,
                        mask_end=mask_end,
                        max_length=max_token_len,
                        exclude_tokens=next_tokens
                        )
                    infer_start = time.time()
                    model_outs = token_model.generate(
                        input_ids,
                        mask_start,
                        mask_end,
                        exclude_tokens=exclude_tokens,
                        temperature=0.9
                        )
                    infer_times += time.time() - infer_start

                    for m in range(len(prompting)-1, -1, -1):
                        n_txt = token_requests[m][0]
                        generated_ids = input_ids[m, mask_start[m]:mask_end[m]].tolist()
                        # print(f'Pos {start_pos[m]}:{end_pos[m]}, Insert/Replace: {tokenizer.decode(texts[n_txt][start_pos[m]:end_pos[m]], skip_special_tokens=True)} -> {tokenizer.decode(generated_ids, skip_special_tokens=True)}')
                        texts[n_txt] = texts[n_txt][:start_pos[m]] + generated_ids + texts[n_txt][end_pos[m]:]

                    token_requests = []

                pos = new_pos
            else:
                pos += get_n(probas_long)
                # print(random.random(), '>', init_prob)
                
        if len(token_requests):
            # print('n_requests', len(token_requests))
            prompting = [get_token_prompt(texts[t_n], start, end, max_length=max_token_len) for t_n, start, end, next_token in token_requests]
            prompts, mask_start, mask_end, start_pos, end_pos, next_tokens= map(list, zip(*prompting))

            input_ids, mask_start, mask_end, exclude_tokens = preproc_token_seqments(
                prompts,
                mask_start=mask_start,
                mask_end=mask_end,
                max_length=max_token_len,
                exclude_tokens=next_tokens
                )
            infer_start = time.time()
            model_outs = token_model.generate(
                input_ids,
                mask_start,
                mask_end,
                exclude_tokens=exclude_tokens,
                temperature=0.9
                )
            infer_times += time.time() - infer_start

            for m in range(len(prompting)-1, -1, -1):
                n_txt = token_requests[m][0]
                generated_ids = input_ids[m, mask_start[m]:mask_end[m]].tolist()
                # print(f'Pos {start_pos[m]}:{end_pos[m]}, Insert/Replace: {tokenizer.decode(texts[n_txt][start_pos[m]:end_pos[m]], skip_special_tokens=True)} -> {tokenizer.decode(generated_ids, skip_special_tokens=True)}')
                texts[n_txt] = texts[n_txt][:start_pos[m]] + generated_ids + texts[n_txt][end_pos[m]:]
            token_requests = []

    texts = tokenizer.batch_decode(texts, skip_special_tokens=True)

    # CHAR LEVEL
    for tn in range(len(texts)):
        text_w_mistakes, n_mistakes = gen_ortho_mistakes(texts[tn], literacy_rate=ortho_mod_rate)
        texts[tn] = text_w_mistakes
        ortho_changes += n_mistakes

        current_char_probas = modify_char_probas(char_probas)
        init_prob = char_mod_rate ** (1. + random.random() * 1.5)
        # print(f'char mod_rate={init_prob}')
        # print(init_prob)
        mod_probs = np.random.rand(4) * np.array([0.9, 0.03, 0.03, 0.04]) # replace, remove, insert, swap
        mod_probs = mod_probs/mod_probs.sum()
        # texts[tn] = re.sub(r' +', ' ', texts[tn])
        # literacy_rate = 1. + random.random()*3.
        pos = 0
        while pos < len(texts[tn]):
            if texts[tn][pos:pos+5] == '[UNK]':
                pos += 5
            if random.random() < init_prob:
                char_changes += 1
                mod_func = random.choices(
                    [replace_chars, remove_chars, insert_chars, swap_chars],
                    weights=mod_probs, k=1
                )[0]
                orig_text = copy.deepcopy(texts[tn])
                texts[tn], new_pos = mod_func(texts[tn], pos)
                if mod_func in [replace_chars, insert_chars]:
                    if (mod_func == replace_chars) and (new_pos == pos+1) and (texts[tn] == orig_text) and (random.random() < 0.2):
                        texts[tn] = replace_1char(texts[tn], pos, current_char_probas)
                        # print(orig_text[pos-5:pos+6], '-->>', texts[tn][pos-5:pos+6])
                    else:
                        if mod_func == replace_chars:
                            next_char = bytetokenizer.token_to_id(texts[tn][pos])
                            if not next_char:
                                next_char = 0
                        else:
                            next_char = 0
                        modification_requests.append((tn, pos, new_pos, next_char))
                # print(mod_func)
            
            if len(modification_requests) == max_char_requests:
                # print('modification_requests: ', len(modification_requests))
                prompting = [get_char_prompt(texts[tn], start, end, max_length=max_char_len) for tn, start, end, next_char in modification_requests]
                prompts, mask_start, mask_end, start_pos, end_pos, next_chars = map(list, zip(*prompting))
                if max(mask_end) > max_char_len:
                    print('min mask start', min(mask_start))
                    print('max mask end', max(mask_end))

                input_ids, mask_start, mask_end, exclude_tokens = preproc_char_seqments(
                    prompts,
                    mask_start=mask_start,
                    mask_end=mask_end,
                    mask_id=bytetokenizer.get_vocab_size(),
                    exclude_tokens=next_chars
                    )
                infer_start = time.time()
                # print(f'modification_requests: {len(modification_requests)}')
                # print(f'prompts: {prompts.shape}')
                # print(f'mask_start: {len(mask_start)}')
                # print(f'mask_end: {len(mask_end)}')
                # print(f'next_chars: {len(next_chars)}')

                model_outs = char_model.generate(
                    input_ids,
                    mask_start,
                    mask_end,
                    exclude_tokens=exclude_tokens,
                    temperature=0.9
                    )
                char_infer_times += time.time() - infer_start
                
                for m in range(len(prompting)):
                    generated_ids = input_ids[m, mask_start[m]:mask_end[m]]
                    generated_chars = bytetokenizer.decode(generated_ids.tolist())

                    assert len(generated_chars) == end_pos[m] - start_pos[m]
                    texts[tn] = texts[tn][:start_pos[m]] + generated_chars + texts[tn][end_pos[m]:]

                modification_requests = []
                pos = new_pos
            else:
                pos += get_n(probas_long)
            
    if len(modification_requests):
        # print('modification_requests: ', len(modification_requests))
        prompting = [get_char_prompt(texts[tn], start, end, max_length=max_char_len) for tn, start, end, next_char in modification_requests]
        prompts, mask_start, mask_end, start_pos, end_pos, next_chars = map(list, zip(*prompting))
        if max(mask_end) > max_char_len:
            print('min mask start', min(mask_start))
            print('max mask end', max(mask_end))

        input_ids, mask_start, mask_end, exclude_tokens = preproc_char_seqments(
            prompts,
            mask_start=mask_start,
            mask_end=mask_end,
            mask_id=bytetokenizer.get_vocab_size(),
            exclude_tokens=next_chars
            )
        infer_start = time.time()
        # print(f'modification_requests: {len(modification_requests)}')
        # print(f'prompts: {prompts.shape}')
        # print(f'mask_start: {len(mask_start)}')
        # print(f'mask_end: {len(mask_end)}')
        # print(f'next_chars: {len(next_chars)}')
        model_outs = char_model.generate(
            input_ids,
            mask_start,
            mask_end,
            exclude_tokens=exclude_tokens,
            temperature=0.9
            )
        char_infer_times += time.time() - infer_start
        for m in range(len(prompting)):
            generated_ids = input_ids[m, mask_start[m]:mask_end[m]]
            generated_chars = bytetokenizer.decode(generated_ids.tolist())

            assert len(generated_chars) == end_pos[m] - start_pos[m]
            texts[tn] = texts[tn][:start_pos[m]] + generated_chars + texts[tn][end_pos[m]:]
        modification_requests = []
        pass

    texts = [text.replace('\n', '[NEWLINE]') for text in texts]
    
    # print(f'char changes: {char_changes}')
    # print(f'ortho changes: {ortho_changes}')
    # print('token changes:', token_changes)#, 'tn:', tn

    # print('translate times:', translate_times)
    # print('token infer time:', infer_times)
    # print('char infer:', char_infer_times)
    # print('tokens:', time.time() - start_time)

    return texts, (token_changes, ortho_changes, char_changes)
