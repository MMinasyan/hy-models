import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from preproc import clean4model, postprocess, armchars, clean_bytes, armchars_l
import re
import random
import difflib
from time import time
import socket
import sys
import torch.distributed as dist
import pickle

dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)  # Set this process's GPU
device = torch.device(f'cuda:{local_rank}')

print(f"Process {local_rank} is using {device}") 

def is_hy(text):
    remove_chars = str.maketrans('', '', ''.join(set(text) - armchars))
    filtered_text = text.translate(remove_chars)
    percentage = (len(filtered_text) / len(text)) if text else 0
    return percentage >= 0.6

model_name = "facebook/nllb-200-distilled-600M"
en_tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=256)
hy_tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="hye_Armn", max_len=256)#token=True, 
nllb = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    max_length=256,
    # device_map = 'auto'
    ).to(device)

nllb = torch.nn.parallel.DistributedDataParallel(nllb, device_ids=[local_rank], output_device=local_rank)

# pipe_madlad = pipeline(
#     "translation",
#     model="google/madlad400-3b-mt",
#     device=device,
#     torch_dtype=torch.float16,
#     max_length=256,
#     )

def get_len(text):
    encoded = hy_tokenizer.encode(text)
    return len(encoded)-2

def verify_bt(text1, text2, treshold=0.5):
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return (matcher.ratio() >= treshold) and (len(text1) / len(text2) < 1.6)

armchars_str = ''.join(list(armchars_l))

def fix_quotes(text):
    pattern = r'(^|[ \n])"([^"]*?[' + armchars_str + r'][^"]*?)"([ ,.:՝…%°?!)\]\}]|$)'
    pattern = re.compile(pattern, re.IGNORECASE)
    return re.sub(pattern, r'\1«\2»\3', text)

def split_paragraph(text, max_len):
    groups = []
    segments = re.split(r'(?<=: )', text)
    group = ''
    group_len = 0
    for segment in segments:
        segment_len = get_len(segment)
        if segment_len > max_len:
            if group_len:
                groups.append(group)
            group = ''
            group_len = 0
            sub_segments = re.split(r'(?<=, )', segment)
            for sub_segment in sub_segments:
                sub_len = get_len(sub_segment)
                if sub_len > max_len:
                    if group_len:
                        groups.append(group)
                    groups.append(sub_segment)
                    group = ''
                    group_len = 0
                elif group_len + sub_len > max_len:
                    groups.append(group)
                    group_len = sub_len
                    group = sub_segment
                else:
                    group_len += sub_len
                    group += sub_segment

            if group_len > 0:
                groups.append(group)
            group = ''
            group_len = 0
        else:
            if group_len + segment_len > max_len:
                if group_len > 0:
                    groups.append(group)
                group_len = segment_len
                group = segment
            else:
                group_len += segment_len
                group += segment
    if group_len > 0:
        groups.append(group)
    return groups

# def use_madlad(batch):
#     batch = ['<2en> ' + text for text in batch]
#     # start = time()
#     back_prompt = pipe_madlad(batch, num_beams=3)
#     back_prompt = ['<2hy> ' + item['translation_text'] for item in back_prompt]
#     back_translated = pipe_madlad(back_prompt, num_beams=3)
#     back_translated = [clean_bytes(item['translation_text']) for item in back_translated]
#     # print(back_translated)
#     back_translated = split_endings(back_translated)[0]
#     # print(f'{round(time()-start, 2)}s for {len(batch)} texts')
#     return back_translated

def split_endings(texts):
    pattern = r'[\n :.,;…|]*$'
    endings = [re.search(pattern, text).group() for text in texts]
    texts = [re.sub(pattern, '', text) for text in texts]
    return texts, endings

def use_nllb(batch, model, num_beams=3, max_len=200):
    # start = time()
    inputs = hy_tokenizer(batch, return_tensors="pt", padding='longest').to(device)
    with torch.no_grad():
        translated_tokens = model.module.generate(
        **inputs, num_beams=num_beams, length_penalty=1.2, forced_bos_token_id=hy_tokenizer.lang_code_to_id["eng_Latn"], max_length=max_len
        )
    back_prompt = en_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    # print(back_prompt[0])
    inputs = hy_tokenizer(back_prompt, return_tensors="pt", padding='longest').to(device)
    with torch.no_grad():
        translated_tokens = model.module.generate(
        **inputs, num_beams=num_beams, length_penalty=1.2, temperature=0.85, do_sample=True, forced_bos_token_id=hy_tokenizer.lang_code_to_id["hye_Armn"], max_length=max_len
        )
    back_translated = hy_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    back_translated = [clean_bytes(bt_text) for bt_text in back_translated]
    # print(back_translated)
    back_translated = split_endings(back_translated)[0]
    back_translated = [fix_quotes(t) for t in back_translated]
    # print(f'{round(time()-start, 2)}s for {len(batch)} texts')
    return back_translated


def back_translate_texts(texts, model1, model2=None, batch_size=16, max_len=200, proba=0.1):
    groups = {}
    all_texts_to_translate = []
    all_keys_for_translation = []

    # Iterate over each text and its index
    for text_id, text in enumerate(texts):
        text_translate = random.random() < proba
        text = postprocess(text)
        paragraphs = re.split(r'(?<=\n)(?=[^\n])', text)
        groups_str = [group for paragraph in paragraphs for group in split_paragraph(paragraph, max_len=130)]
        groups_str, group_end = split_endings(groups_str)

        for i, (group_str, end) in enumerate(zip(groups_str, group_end)):
            group_dict = {
                'string': group_str,
                'ending': end,
                'translate': 'none',
                'bt': group_str  # Default to original string
            }
            group_len = get_len(group_str)
            if group_len > 4 and group_len < max_len and is_hy(group_str) and text_translate:
                group_dict['translate'] = 'model1'
                all_texts_to_translate.append(group_str)
                all_keys_for_translation.append((text_id, i))

            groups[(text_id, i)] = group_dict

    # print(f'n_groups: {len(groups)}')
    # Process translations in batches
    translated_texts = []
    for i in range(0, len(all_texts_to_translate), batch_size):
        batch = all_texts_to_translate[i:i + batch_size]
        translated_texts.extend(use_nllb(batch, model1, num_beams=3))

    re_translate_texts = []
    re_keys_for_translation = []
    # Assign back-translated texts to the respective groups
    for key, translated_text in zip(all_keys_for_translation, translated_texts):
        if verify_bt(groups[key]['string'], translated_text, treshold=0.4):
            groups[key]['bt'] = translated_text
        else:
            # groups[key]['bt'] = groups[key]['string']  # Fall back to original if not verified
            if model2 != None:
                groups[key]['translate'] = 'model2'
                re_translate_texts.append(groups[key]['string'])
                re_keys_for_translation.append(key)
            else:
                groups[key]['bt'] = groups[key]['string']
                re_keys_for_translation.append(key)
                # print(f'\nORIGINAL: {groups[key]["string"]}\n BTLATED: {translated_text}')
    # print(f'Failed to verify with model1: {len(re_keys_for_translation)} / {len(all_texts_to_translate)}')
    if model2 != None:
        re_translated_texts = []
        for i in range(0, len(re_translate_texts), batch_size):
            batch = re_translate_texts[i:i + batch_size]
            re_translated_texts.extend(use_nllb(batch, model2))

        skipped = 0
        # Assign re-translated texts to the respective groups
        for key, re_translated_text in zip(re_keys_for_translation, re_translated_texts):
            if verify_bt(groups[key]['string'], re_translated_text):
                groups[key]['bt'] = re_translated_text
            else:
                groups[key]['bt'] = groups[key]['string']
                skipped += 1
        # print(f'Failed to verify with model2: {skipped}')
    # Compile all texts from the groups
    compiled_texts = [''] * len(texts)
    for (text_id, i), group in groups.items():
        compiled_texts[text_id] += group['bt'] + group['ending']
    return [clean4model(compiled_text, use_pre_clean=True) for compiled_text in compiled_texts]#, groups

if local_rank == 0:
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    port = 9999
    serversocket.bind((host, port))
    serversocket.listen(1)
    print("Server is ready and listening.")
else:
    print(f'GPU {local_rank} is waiting for tasks.')

while True:
    if local_rank == 0:
        clientsocket, addr = serversocket.accept()
        data_chunks = []
        while True:
            chunk = clientsocket.recv(16384)
            if not chunk:
                break
            data_chunks.append(chunk)
        data_received = pickle.loads(b''.join(data_chunks))
    
    # Broadcast data to all GPUs
    data_received = [data_received] if local_rank == 0 else [None]
    dist.broadcast_object_list(data_received, src=0)
    data_received = data_received[0]

    # Split the data to distribute to GPUs
    num_gpus = dist.get_world_size()
    chunk_size = len(data_received) // num_gpus
    chunks = [data_received[i:i + chunk_size] for i in range(0, len(data_received), chunk_size)]
    if len(data_received) % num_gpus != 0:  # Handling remainder
        chunks[-1].extend(data_received[num_gpus * chunk_size:])

    # print(f"GPU {local_rank} is processing {len(chunks[local_rank])} samples.")
    # Each GPU processes its chunk of data
    local_results = back_translate_texts(chunks[local_rank], nllb, batch_size=72)

    # Gather results from all GPUs
    all_results = [None] * num_gpus
    dist.all_gather_object(all_results, local_results)

    if local_rank == 0:
        final_results = [item for sublist in all_results for item in sublist]
        result_bytes = pickle.dumps(final_results)
        clientsocket.sendall(result_bytes)
        clientsocket.close()