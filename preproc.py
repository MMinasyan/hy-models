import re
from urllib.parse import unquote

bad_chars_dict = {
    '\r': '\n',
    '：': ':',
    '∶': ':',
    '։':':',
    '-': '-',
    '֊': '-',
    '−': '-',
    '‒': '-',
    '―': '—',
    '–': '—',
    '‚': ',',
    '՚': '’',
    '¹': '^1',
    '²': '^2',
    '³': '^3',
    '´': '՛',
    '′': '՛',
    '½': '1/2',
    '¦': '|',
    'ا': '|',
    '∙': '-',
    '‑': '-',
    'ﬕ': 'մի',
    'և': 'եւ',
    '⁰': '°',
    '˚': '°',
    '¼': '1/4',
    'ﬔ': 'մե',
    '∗': '*',
    '\xa0': '',
    '•': '-',
    '◦': '-',
    '’': "'",
    '”': '"',
    '“': '"',
    '․': '.',
    '`': '՝'
}

bad_chars_table = str.maketrans(bad_chars_dict)

def clean_bytes(text):
    text = ''.join([ch for ch in text if ch.isprintable() or ch.isspace()]).replace('&nbsp;', ' ')
    text = re.sub(r'[^\S\n\t]+', ' ', text)
    text = text.translate(bad_chars_table)
    return re.sub(r' +', ' ', text)

def move_intonation(text):
    pattern = re.compile(r'([՜՛՞])([Ա-Ֆա-ֆ]+)')
    result = re.sub(pattern, lambda m: m.group(2) + m.group(1), text)
    return result

def sep_marks(text):
    first_set = r'[\[\]\(\),՝:՜՛՞«»—%°?!{}\/]'
    second_set = r'((?:-+)|(?:\.+)|(?:_+))'
    text = re.sub(first_set, r' \g<0> ', text)
    text = re.sub(second_set, r' \1 ', text)
    text = re.sub(r' +', ' ', text)
    text= text.replace('[ NEWLINE ]', '[NEWLINE]')
    return text

def fix_urls(text):
    url_pattern = r'(https?://[^\s]+)'
    return re.sub(url_pattern, lambda match: unquote(match.group(0)), text)


armchars = 'աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆ'
armchars = set(armchars + armchars.upper())
ruschars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
ruschars = set(ruschars + ruschars.upper())
latinchars = 'abcdefghijklmnopqrstuvwxyz'
latinchars = set(latinchars + latinchars.upper())
digits = set('0123456789')

primary_marks = {':', '.', ',', '՝', '…'}
intonation_marks = {'՞', '՛', '՜'}
inclusion_marks = {'(', ')', '[', ']', '{', '}', '«', '»'}
other_marks = {'-', '—', '_', '%', '°', '\\', '/', '?', '!', '|', "'", '"', ';', '&', '#', '@', '$', '§', '№',\
               '↑', '→', '¿', '€', '¶', '£', '©', '~', '֏'}
math = {'+', '=', '>', '<', '*', '·', '^', '×', '≥', '≤', '±', '÷', 'Ø', '⊕', 'µ', 'α', 'σ', 'β', 'ν', 'ε', 'φ', 'δ', 'π', 'η',\
        '∑', '∆', '≈'}
marks = primary_marks.union(intonation_marks, inclusion_marks, other_marks, math)
alphanumerics = armchars.union(latinchars, ruschars).union(digits)


def pre_clean(text, replace_newline=False):
    text = clean_bytes(text)
    text = text.replace('o', '-').replace('●', '-')
    text = re.sub(r'\n([ա-ֆ])\. ', r'\n\t\1. ', text)
    text = re.sub(r'\n([ա-ֆ])\) ', r'\n\t\1) ', text)
    text = re.sub(r' +', ' ', text)
    text = text.replace('\n ', '\n').replace(' \n', '\n') 
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'-----+', lambda m: ' '.join(['-----' for _ in range(len(m.group(0)) // 5)]), text)
    text = re.sub(r'\.{5,}', lambda m: ' '.join(['.....' for _ in range(len(m.group(0)) // 5)]), text)
    text = re.sub(r'_{5,}', lambda m: ' '.join(['_____' for _ in range(len(m.group(0)) // 5)]), text)
    text = re.sub(r'\n[-_* ]{3,}\n', '\n-----\n', text).strip()
    
    text = re.sub(r'(?<!\n)[ ]*\n[ ]*(?!\n|-|\||\d+\.|\t)', r'\n\n', text) # separate regular lines by double new-line
    text = re.sub(r'(?:\n^\n+)\n[ ]*(?!(?:-|—|\||\d+\.|\n))', '\n\n', text)

    text = re.sub(r'(\n[ ]*\|[^\n]*\n)(?![\n|\|])', r'\1\n', text) # add newline after table
    text = re.sub(r'(\n[^\|\n]+\n)(?=\|)', r'\1\n', text) # add newline before table

    text = re.sub(r'(\n[ ]*\- [^\n]*\n)(?![\n|\-])', r'\1\n', text) # add newline after unordered list
    text = re.sub(r'(\n[^\-\n][^\n]+\n)(?=\- )', r'\1\n', text) # add newline before unordered list

    text = re.sub(r'(\n[ ]*\t[^\n]*\n)(?![\n|\t])', r'\1\n', text) # add newline after indented lines
    text = re.sub(r'(\n[^\n\t]+\n)(?=\t)', r'\1\n', text) # add newline before indented lines

    text = re.sub(r'(?<=\n)[ ]*\| +\|[ ]*\n', '', text)

    text = text.replace('&lt;&lt;', '«').replace('&gt;&gt;', '»')
    text = text.replace('<<', '«').replace('>>', '»')
    text = text.replace('\uf6b6', '«').replace('\uf6b7', '»')
    
    text = text.replace('&amp;', '&')
    text = re.sub(r'[ ]*\n[ ]*', '\n', text)

    if replace_newline==True:
        text = text.replace('\n', '[NEWLINE]')
    text = fix_urls(text)
    return re.sub(r' +', ' ', text)

def clean4model(text, use_pre_clean=False):
    s = str(text)
    if use_pre_clean:
        s = pre_clean(s)
    else:
        s = clean_bytes(s)
        s = re.sub(r' +', ' ', s)
    s = s.replace('<<', '«')
    s = s.replace('>>', '»')
    s = re.sub(r'(?<=[Ա-Ֆա-ֆ])~', '՜', s)
    s = re.sub(r'(?<!\.)\.{3}(?!\.)', '…', s)
    s = move_intonation(s)
    s = re.sub(r'( *<br> *)+', ' <br> ', s)
    s = re.sub(r'( *<br/> *)+', ' <br/> ', s)
    s = sep_marks(s)
    s = s.strip()
    return s

def encode_2d(sequences, tokenizer, max_len_w, max_len=None):
    sequences = sequences.split(' ')
    encoded = [tokenizer.encode(sequence, add_special_tokens=False).ids[:max_len_w] for sequence in sequences]
    encoded = [word + [tokenizer.token_to_id('[PAD]')]*max(max_len_w - len(word), 0) for word in encoded]
    if max_len is not None:
        encoded = encoded[:max_len]
        encoded += [[tokenizer.token_to_id('[PAD]')] * max_len_w ]* max(max_len - len(sequences), 0)
    return encoded


armchars_l = {c.lower() for c in armchars}
emph_consonants = ''.join(list(armchars_l.difference({'ա','ո','օ','ե','է','ի'})))
emph_vowels = 'աոօեէի'
emph_first_words = r'գրեթե|միմիայն|միգուցե|միթե|որերորդ|որեւէ|գոնե|գուցե|մանավանդ|նույնիսկ|նույնպես|նույնքան|նույնպիսի'
emph_first_words += r'|' + r'|'.join([fr'{ss}երորդ' for ss in 'հինգ|վեց|յոթ|ութ|ինն|տասն|քսան|երեսուն|քառասուն|հիսուն|վաթսուն|յոթանասուն|ութսուն|ինսուն|հարյուր|հազար|միլիոն|միլիարդ'.split('|')])

def restore_intonation(text):
    first_int = r'('+emph_first_words+') ([՜՛՞])'
    first_int = re.compile(first_int, re.IGNORECASE)
    text = re.sub(first_int, lambda m: re.sub(f'([{emph_vowels}])', r'\1' + m.group(2), m.group(1), 1), text)

    text = re.sub(r'իհարկե ([՜՛՞])', r'իհա\1րկե', text)
    pattern = re.compile(fr'([{emph_vowels}])([{emph_consonants}]+)? ([՜՛՞])')
    text = re.sub(pattern, r'\1\3\2', text)
    return text

def restore_punctuation(text):
    text = re.sub(r'(\d) : (\d{2})', r'\1:\2', text)
    text = re.sub(r'(\d) \. (\d)', r'\1.\2', text)
    text = re.sub(r' ([\]\})»,.:՝…%°?!])', r'\1', text)
    text = re.sub(r'([\[\{«\(]) ', r'\1', text)
    text = re.sub(r'(?<!^)(?<![ա-ֆ]{2})(?<![\]\})»,.:՝…%°?!])(?<!\[NEWLINE\]) (-|—) (?=ը|ն|ի|ո|ա|ր)', r'\1', text)
    text = re.sub(r'(?:(?<=[Ա-Ֆ]\.)|(?<=[\]\})».%°?!])) (-|—) (?=ը\b|ն\b|ի\b|ին|ից|ով|ու|րդ)', r'\1', text)
    text = re.sub(r' / ', '/', text)
    return text

def postprocess(text, replace_newline=True):
    text = restore_intonation(text)
    text = restore_punctuation(text)
    if replace_newline==True:
        text = text.replace('[NEWLINE]', '\n')
        text = re.sub(r'( +)\n( +)?|\n( +)', '\n', text)
    return text.strip()
