import re
import random
import json
import numpy as np

from preproc import armchars


armchars_lower = ''.join([c for c in armchars if c.islower()])
armchars_upper = armchars_lower.upper()
vowels = {'ա', 'ե', 'է', 'ի', 'ո', 'ւ', 'օ'}
vowels_l = ''.join(vowels)
vowels_u = ''.join(vowels).upper()
cons = set(armchars_lower).difference(vowels).difference({'ը'})
cons_l = ''.join(cons)
cons_u = ''.join(cons).upper()

def use_case(input, output):
    if input.isupper():
        return output.upper()
    else:
        return output.lower()

def use_cases(input, output):
    out_chars = []
    upper_case = False
    for i in range(len(output)):
        if i < len(input):
            upper_case = input[i].isupper()
        next_char = output[i].upper() if upper_case else output[i]
        out_chars.append(next_char)
    return ''.join(out_chars)

def apply_patterns(text, patterns, probability=0.5):
    replacements = []
    for pattern, replacement in patterns:
        for match in pattern.finditer(text):
            if random.random() < probability:
                start, end = match.start(), match.end()
                # Directly applying replacement, accounting for potential backreferences
                matched_text = text[start:end]
                replacement_text = re.sub(pattern, replacement, matched_text)
                replacements.append((start, end, replacement_text))
    replacements.sort(key=lambda x: x[0], reverse=True)
    for start, end, replacement in replacements:
        text = text[:start] + replacement + text[end:]
    return text

def ye_samecase(x):
    return use_case(x, 'ե')

def last_e_samecase(x):
    return x[:-1] + use_case(x[-1], 'է')

def e_samecase(x):
    return use_case(x, 'է')

def o_samecase(x):
    return use_case(x[0], 'օ')

def vo_samecase(x):
    return use_case(x[0], 'ո')

def vvo_samecases(x):
    if len(x)>1:
        return use_case(x[0], 'վ') + use_case(x[1], 'ո')
    else:
        return use_case(x, 'վո')

def removematch(x):
    return ''
def y_first_seq_samecase(x):
    y = use_case(x[0], 'ը')
    if x[1].lower():
        return y + x[0] + x[1:]
    else:
        return y + x
def y_samecase_first(x):
    return use_case(x, 'ը') + x

def dzaynakap_samecase_first(x):
    return use_case(x, 'յ') + x

def v_samecase(x):
    return use_case(x, 'վ') + x

def vyun_samecase(x):
    return use_case(x, 'ւ') + x

def i_samecase(x):
    return use_case(x, 'ի')

def dzaynakap_samecase(x):
    return use_case(x, 'յ')

def ph_samecase(x):
    return use_case(x, 'փ')

def p_samecase(x):
    return use_case(x, 'պ')

def b_samecase(x):
    return use_case(x, 'բ')

def khp_samecase(x):
    return use_case(x[0], 'խ') + use_case(x[1], 'պ')

def ghp_samecase(x):
    return use_case(x[0], 'ղ') + use_case(x[1], 'պ')

def khb_samecase(x):
    return use_case(x[0], 'խ') + use_case(x[1], 'բ')

def q_samecase(x):
    return use_case(x, 'ք')

def k_samecase(x):
    return use_case(x, 'կ')

def t_samecase(x):
    return use_case(x, 'տ')

def th_samecase(x):
    return use_case(x, 'թ')

def d_samecase(x):
    return use_case(x, 'դ')

def w_test_samecase(x):
    return use_cases(x, 'տեստ')

def w_tհestհ_samecase(x):
    return use_cases(x, 'թեսթ')
    
def tz_samecase(x):
    return use_case(x, 'ց')

def ts_samecase(x):
    return use_case(x, 'ծ')

def dz_samecase(x):
    return use_case(x, 'ձ')

def khtz_samecase(x):
    return use_cases(x, 'խց')
def ch_samecase(x):
    return use_case(x, 'չ')

def j_samecase(x):
    return use_case(x, 'ջ')

def tch_samecase(x):
    return use_case(x, 'ճ')

def khch_samecase(x):
    return use_cases(x, 'խչ')

def ghch_samecase(x):
    return use_cases(x, 'ղչ')

def ghj_samecase(x):
    return use_cases(x, 'ղջ')

def sh_samecases(x):
    return use_cases(x, 'շ')
    
def kh_samecases(x):
    return use_cases(x, 'խ')

def gh_samecases(x):
    return use_cases(x, 'ղ')
    
def r_samecases(x):
    return use_cases(x, 'ր')

def rh_samecases(x):
    return use_cases(x, 'ռ')

def m_samecases(x):
    return use_cases(x, 'մ')

def n_samecases(x):
    return use_cases(x, 'ն')

def v_samecase(x):
    return use_cases(x, 'վ')

def f_samecase(x):
    return use_cases(x, 'ֆ')

def add_h_samecase(x):
    return x + use_cases(x[-1], 'հ')

def add_i_samecase(x):
    return x + use_cases(x[-1], 'ի')

def add_h_before_samecase(x):
    return use_cases(x, 'հ'+x.lower())

def add_same_samecase(x):
    return x * 2

def add_n_samecase(x):
    return x + use_case(x, 'ն')

def add_q_samecase_first(x):
    return use_case(x, 'ք') + x

def add_comma(x):
    return ',' + x

def add_midpoint(x):
    return '.' + x

def add_but(x):
    return '՝' + x

def add_question(x):
    return '՞' + x

def add_emphasis(x):
    return '՛' + x

def add_exclamation(x):
    return '՜' + x

def add_endpoint(x):
    return ':' + x

def write_latquest(x):
    return '?'

def write_latquest_before_end(x):
    return x[2:-1] + '? ' + x[-1]

def write_latquest_after_end(x):
    return x[2:-1] + x[-1] + ' ?'

def write_latquest_at_end(x):
    return x[2:-1] + '?'

def write_quest_before_end(x):
    return x[2:-1] + '՞ ' + x[-1]

def write_quest_after_end(x):
    return x[2:-1] + x[-1] + ' ՞'

def write_quest_at_end(x):
    return x[2:-1] + '՞'

def write_latemph(x):
    return '!'

def write_latemph_at_end(x):
    return x[2:-1] + '!'

def swap_samecases(x):
    return use_case(x[0], x[1]) + use_case(x[1], x[0])

def move_first2end_samecase(x):
    moved = x[1:] + x[0]
    result = ''.join(moved[i].upper() if x[i].isupper() else moved[i].lower() for i in range(len(moved)))
    return result

def hy_lower(x):
    return x.replace('ԵՎ', 'եւ').lower()

def hy_upper(x):
    return x.replace('եւ', 'եվ').upper()

def cap_from_upper(x):
    return x[0] + hy_lower(x[1:])

def cap_from_lower(x):
    if x.startswith('եւ'):
        return 'Եվ' + x[2:]
    else:
        return hy_upper(x[0]) + x[1:]

def cap_allwords(x):
    return re.sub(r'\b\S+\b', lambda match: cap_from_lower(match.group()), x)

def lower_text_random(txt):
    txt = txt.split(' ')
    words = []
    first = True
    for w in txt:
        if w:
            if first:
                if random.random()<0.01:
                    words.append(w)
                else:
                    words.append(cap_from_upper(w))
            elif random.random()<0.1:
                words.append(cap_from_upper(w))
            elif random.random()<0.01:
                words.append(w)
            else:
                words.append(hy_lower(w))
            first = False
        else:
            words.append(w)
    return ' '.join(words)

def write_vorpeszi(x):
    return use_cases(x, 'որպեսզի')

def add_a_samecase(x):
    return use_cases(x, 'ա')

def rand_move_quest(x):
    num_quest = x.count('՞')
    for q_n in range(num_quest):
        x = x.replace('՞', '')
        pos = random.randint(0, len(x))
        x = x[:pos] + '՞' + x[pos:]
    return x

def separate_veratsvel(x):
    substrings = [ss for ss in re.split('\\b', x) if ss]
    if len(substrings) == 1:
        return 'վեր ' + substrings[0][3:]
    else:
        main, space, aux = substrings
        return'վեր' + space + aux + space + main[3:]

def add_u_samecase_end(x):
    return x + use_case(x[-1], 'ու')

patterns_dicts = []

patterns_dicts.append(
    {'pattern': r"(?:(?<=\b)է|(?<=\bչ)է)",
     'ignore_case': True,
     'replacement': [(ye_samecase, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r"\Bէ(?=ա|ջ|կր|ներգ)",
     'ignore_case': True,
     'replacement': [(ye_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"\bչ?ե(?=մ|ս|ն|ք)",
     'ignore_case': True,
     'replacement': [(last_e_samecase, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r"\Bե\b",
     'ignore_case': True,
     'replacement': [(e_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=եւ|եվ)է\b",
     'ignore_case': True,
     'replacement': [(e_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"\Bօ",
     'ignore_case': True,
     'replacement': [(vo_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"\bո\B(?!վ)?",
     'ignore_case': True,
     'replacement': [(vvo_samecases, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"\bվո",
     'ignore_case': True,
     'replacement': [(vo_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"\bո(?=վ)",
     'ignore_case': True,
     'replacement': [(o_samecase, 0.13)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=\B[{vowels_l}])ո",
     'ignore_case': True,
     'replacement': [(vvo_samecases, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"\Bվո(?=ր)",
     'ignore_case': True,
     'replacement': [(vo_samecase, 0.06)]}
)
patterns_dicts.append(
    {'pattern': r"ը(?=ղ|մ|ն|ստ)",
     'ignore_case': True,
     'replacement': [(removematch, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<![{vowels_l+'ը'}])(?:սկ|ստ|սփ|սթ|սք|զբ|զգ|շտ|շպ)",
     'ignore_case': True,
     'replacement': [(y_first_seq_samecase, 0.03)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=([{cons_l}]))[{cons_l}]",
     'ignore_case': True,
     'replacement': [(y_samecase_first, 0.001)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[աոօ])յ(?=[{vowels_l+'ը'}])",
     'ignore_case': True,
     'replacement': [(removematch, 0.15)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[էեիւ])[{vowels_l+'ը'}]",
     'ignore_case': True,
     'replacement': [(dzaynakap_samecase_first, 0.15)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=(մենա|ենթա|հակա|կիսա))[{vowels_l+'ը'}]",
     'ignore_case': True,
     'replacement': [(dzaynakap_samecase_first, 0.05)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=(նո|հա|գո|գո|թե|պե))յ(?=[{vowels_l+'ը'}])",
     'ignore_case': True,
     'replacement': [(removematch, 0.05)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=ե)ւ",
     'ignore_case': True,
     'replacement': [(v_samecase, 0.13)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ե)վ",
     'ignore_case': True,
     'replacement': [(vyun_samecase, 0.13)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=[եէիւ])յ(?=[աոեի])",
     'ignore_case': True,
     'replacement': [(removematch, 0.13)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<![{vowels_l}])յ(?=[աոեի])",
     'ignore_case': True,
     'replacement': [(i_samecase, 0.05)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{vowels_l}])յ(?![{vowels_l}])",
     'ignore_case': True,
     'replacement': [(i_samecase, 0.1), (removematch, 0.03)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<![{vowels_l}])ի(?=[աո])",
     'ignore_case': True,
     'replacement': [(dzaynakap_samecase, 0.12), (ye_samecase, 0.03)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<![{vowels_l}])ե(?=աո)",
     'ignore_case': True,
     'replacement': [(dzaynakap_samecase, 0.08), (i_samecase, 0.07)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<![{vowels_l}])ի(?=ե)",
     'ignore_case': True,
     'replacement': [(dzaynakap_samecase, 0.12)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<![{vowels_l}])յ(?=ուն)",
     'ignore_case': True,
     'replacement': [(removematch, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[եէ])ի",
     'ignore_case': True,
     'replacement': [(dzaynakap_samecase, 0.04)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=ր)բ",
     'ignore_case': True,
     'replacement': [(ph_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(ար|եր|փր))փ",
     'ignore_case': True,
     'replacement': [(b_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(խա|շա|ամ|գա))բ",
     'ignore_case': True,
     'replacement': [(ph_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=\bի)բ(?=ր)",
     'ignore_case': True,
     'replacement': [(ph_samecase, 0.15)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(ակո|երո))բ",
     'ignore_case': True,
     'replacement': [(ph_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{vowels_l}])ղբ",
     'ignore_case': True,
     'replacement': [(khp_samecase, 0.08), (ghp_samecase, 0.07), (khb_samecase, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=հա)փ",
     'ignore_case': True,
     'replacement': [(p_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=հա)պ",
     'ignore_case': True,
     'replacement': [(ph_samecase, 0.08)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=(եր|ար|.\Bմ|.\Bն))գ",
     'ignore_case': True,
     'replacement': [(q_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(ւոր|ՎՈՐ|Մար|ՄԱՐ|Սար|ՍԱՐ|Պար|ՊԱՐ))գ",
     'ignore_case': False,
     'replacement': [(q_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(վա|զա|ու|թա|ծա|ծե|րա|հա|հո|ձա|ձի|ճի|րո|շո|տե|ան))գ",
     'ignore_case': True,
     'replacement': [(q_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(է|ի|օ))գ",
     'ignore_case': True,
     'replacement': [(q_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[ղխ])ք",
     'ignore_case': True,
     'replacement': [(k_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[ղխ])կ",
     'ignore_case': True,
     'replacement': [(q_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=վարա)կ",
     'ignore_case': True,
     'replacement': [(q_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{cons_l.replace('ղ', '').replace('', 'խ')}])կ\b",
     'ignore_case': True,
     'replacement': [(q_samecase, 0.07)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[ծչշ])ք",
     'ignore_case': True,
     'replacement': [(k_samecase, 0.1)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=\B..)(?<![էյ])դ\b",
     'ignore_case': True,
     'replacement': [(t_samecase, 0.15)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=\bէ)դ|(?<=\bայ)դ",
     'ignore_case': True,
     'replacement': [(t_samecase, 0.15)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(ար|եր|ւր|բր|դր|որ|վր|դա|ան|խն|են|ըն|ար|եր|ւր|որ|խր|քր|կր|ըն))դ",
     'ignore_case': True,
     'replacement': [(th_samecase, 0.15)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=օ)դ",
     'ignore_case': True,
     'replacement': [(th_samecase, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=օ)թ",
     'ignore_case': True,
     'replacement': [(d_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ո)թ(?=ք)",
     'ignore_case': True,
     'replacement': [(t_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=թուղ)թ",
     'ignore_case': True,
     'replacement': [(t_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=թղ)թ",
     'ignore_case': True,
     'replacement': [(t_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=թա)դ(?=եւոս|եվոս)",
     'ignore_case': True,
     'replacement': [(th_samecase, 0.08)]}
)
patterns_dicts.append(
    {'pattern': r"թեստ",
     'ignore_case': True,
     'replacement': [(w_test_samecase, 0.1), (w_tհestհ_samecase, 0.1)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=(ար|եր|ւր|որ|վր))ձ",
     'ignore_case': True,
     'replacement': [(tz_samecase, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=օ)ձ",
     'ignore_case': True,
     'replacement': [(tz_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[աեիւփ])ղձ",
     'ignore_case': True,
     'replacement': [(khtz_samecase, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(ւր|ար|եր))ց",
     'ignore_case': True,
     'replacement': [(dz_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[խկ])ց(?=կ)",
     'ignore_case': True,
     'replacement': [(ts_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(ար|որ))ծ",
     'ignore_case': True,
     'replacement': [(tz_samecase, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"\Bծ(?=կ)",
     'ignore_case': True,
     'replacement': [(tz_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"\Bծ(?=ք)",
     'ignore_case': True,
     'replacement': [(tz_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"\Bց(?=ք)",
     'ignore_case': True,
     'replacement': [(ts_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=\Bա)ծ(?!ք)",
     'ignore_case': True,
     'replacement': [(tz_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=\Bա)ց(?!ք)",
     'ignore_case': True,
     'replacement': [(ts_samecase, 0.05)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=[աէեո])ջ",
     'ignore_case': True,
     'replacement': [(ch_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=մի)ջ",
     'ignore_case': True,
     'replacement': [(ch_samecase, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=\bի)ջ",
     'ignore_case': True,
     'replacement': [(ch_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ր)ջ",
     'ignore_case': True,
     'replacement': [(ch_samecase, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[րռ])չ",
     'ignore_case': True,
     'replacement': [(j_samecase, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ա)չ",
     'ignore_case': True,
     'replacement': [(j_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"ղջ",
     'ignore_case': True,
     'replacement': [(khch_samecase, 0.09), (ghch_samecase, 0.06)]}
)
patterns_dicts.append(
    {'pattern': r"ղչ",
     'ignore_case': True,
     'replacement': [(khch_samecase, 0.09), (ghj_samecase, 0.06)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=\bի)նչ",
     'ignore_case': True,
     'replacement': [(sh_samecases, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ա)չ(?=ք)",
     'ignore_case': True,
     'replacement': [(sh_samecases, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ո)ճ(?=կ)",
     'ignore_case': True,
     'replacement': [(j_samecase, 0.05), (ch_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ճան)ճ",
     'ignore_case': True,
     'replacement': [(j_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=գո)ճ(?=ի)",
     'ignore_case': True,
     'replacement': [(j_samecase, 0.1)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=ա)ղ(?=[թտ])",
     'ignore_case': True,
     'replacement': [(kh_samecases, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ե)ղ(?=[դտճծ])",
     'ignore_case': True,
     'replacement': [(kh_samecases, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ի)ղ(?=[ճծ])",
     'ignore_case': True,
     'replacement': [(kh_samecases, 0.06)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ու)ղ(?=[տթ])",
     'ignore_case': True,
     'replacement': [(kh_samecases, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ո)ղ(?=[փպ])",
     'ignore_case': True,
     'replacement': [(kh_samecases, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{cons_l}])ղ(?=[{cons_l}])",
     'ignore_case': True,
     'replacement': [(kh_samecases, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[ժոաւբվկ])խ(?=տ)",
     'ignore_case': True,
     'replacement': [(gh_samecases, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ւ)խ(?=ս)",
     'ignore_case': True,
     'replacement': [(gh_samecases, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ա)խ(?=[ճշս])",
     'ignore_case': True,
     'replacement': [(gh_samecases, 0.12)]}
)

patterns_dicts.append(
    {'pattern': fr"(?<=[{cons_l}])ռ(?=[{cons_l}])",
     'ignore_case': True,
     'replacement': [(r_samecases, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{cons_l}])ր(?=[{cons_l}])",
     'ignore_case': True,
     'replacement': [(rh_samecases, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ա)ր(?=համար)",
     'ignore_case': True,
     'replacement': [(rh_samecases, 0.1)]}
)

patterns_dicts.append(
    {'pattern': r"մ(?=[բպփ])",
     'ignore_case': True,
     'replacement': [(n_samecases, 0.08)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{vowels_l}])մ(?=[վֆ])",
     'ignore_case': True,
     'replacement': [(n_samecases, 0.08)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ա)ն(?=[բպփ])",
     'ignore_case': True,
     'replacement': [(m_samecases, 0.08)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=հարա)վ",
     'ignore_case': True,
     'replacement': [(f_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=նա)վ(?=թ)",
     'ignore_case': True,
     'replacement': [(f_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ա)վ(?=տո)",
     'ignore_case': True,
     'replacement': [(f_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=աղ)վ(?=ան)",
     'ignore_case': True,
     'replacement': [(f_samecase, 0.03)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ե)ւ(?=իլ)",
     'ignore_case': True,
     'replacement': [(f_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ա)ֆ(?=ղան)",
     'ignore_case': True,
     'replacement': [(v_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=աս)ֆ(?=ալտ)",
     'ignore_case': True,
     'replacement': [(v_samecase, 0.1)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=աշխար|խոնար)հ",
     'ignore_case': True,
     'replacement': [(removematch, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=աշխա)ր(?!հ)",
     'ignore_case': True,
     'replacement': [(add_h_samecase, 0.15)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=խոր)հ(?=ուրդ)",
     'ignore_case': True,
     'replacement': [(removematch, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=խոր)հ(?=րդ)",
     'ignore_case': True,
     'replacement': [(removematch, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=(ճանապար|արհամար))հ",
     'ignore_case': True,
     'replacement': [(removematch, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=շնոր)հ",
     'ignore_case': True,
     'replacement': [(removematch, 0.12)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=ընդ)հ(?=[{vowels_l}])",
     'ignore_case': True,
     'replacement': [(removematch, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=ըն)դ(?=[{vowels_l}])",
     'ignore_case': True,
     'replacement': [(add_h_samecase, 0.15)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=հով)հ(?=ան)",
     'ignore_case': True,
     'replacement': [(removematch, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=օր)հ(?=ն)",
     'ignore_case': True,
     'replacement': [(removematch, 0.03), (sh_samecases, 0.07)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{vowels_l}])հ(?=ն)",
     'ignore_case': True,
     'replacement': [(removematch, 0.09)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{vowels_l}])հ(?=[{vowels_l}])",
     'ignore_case': True,
     'replacement': [(removematch, 0.09)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=հայ)թ(?=այթ)",
     'ignore_case': True,
     'replacement': [(add_h_samecase, 0.12)]}
)
patterns_dicts.append(
    {'pattern': r"\bարբ[եա]|(?<=\bչ)արբ[եա]",
     'ignore_case': True,
     'replacement': [(add_h_before_samecase, 0.15)]}
)
patterns_dicts.append(
    {'pattern': r"\bարդուկ|(?<=\bչ)արդուկ",
     'ignore_case': True,
     'replacement': [(add_h_before_samecase, 0.15)]}
)

patterns_dicts.append(
    {'pattern': fr"(?<=([{cons_l}]))\1",
     'ignore_case': True,
     'replacement': [(removematch, 0.05), (y_samecase_first, 0.05)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=դո)լ(?=ար)|(?<=լիա)ն(?=ա)|(?<=ու)ղ(?=[{vowels_l}])|(?<=ու)ղ(?=վ)|(?<=օ)պ(?=ո)|(?<=օ)ֆ(?=լայն)|(?<=գորի)լ(?=ա)|(?<=է)լ(?=են)|(?<=ե)լ(?=են)|(?<=է)լ(?=իոթ)|(?<=ֆի)լ(?=իպ)|(?<=պրե)ս(?!ս)",
     'ignore_case': True,
     'replacement': [(add_same_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ի)ն(?=ը)",
     'ignore_case': True,
     'replacement': [(add_same_samecase, 0.15)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=ի)ն(?=սուն)",
     'ignore_case': True,
     'replacement': [(add_same_samecase, 0.15)]}
)

patterns_dicts.append(
    {'pattern': r"(?<!բ)ժշ",
     'ignore_case': True,
     'replacement': [(sh_samecases, 0.1)]}
)

patterns_dicts.append(
    {'pattern': r"(\B[^ն])\B(?=եր\b|եր[ընդս]\b|երի\b|երի[ընդսց]\b|եր(?:ով|ու)\b)",
     'ignore_case': True,
     'replacement': [(add_n_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=\B[^ն])\Bն(?=եր\b|եր[ընդս]\b|երի\b|երի[ընդսց]\b|եր(?:ով|ու)\b)",
     'ignore_case': True,
     'replacement': [(removematch, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=\Bն)\Bն(?=եր\b|եր[ընդս]\b|երի\b|երի[ընդսց]\b|եր(?:ով|ու)\b)",
     'ignore_case': True,
     'replacement': [(removematch, 0.1)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=արտասո)ւ(?=ք)",
     'ignore_case': True,
     'replacement': [(add_n_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=է)ս(?=պրեսո)",
     'ignore_case': True,
     'replacement': [(add_q_samecase_first, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=պատճե)ն",
     'ignore_case': True,
     'replacement': [(removematch, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{vowels_l}])([ռր]յ|յ[ռր])(?=[{vowels_l}])",
     'ignore_case': True,
     'replacement': [(swap_samecases, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{cons_l}])յու[{cons_l}](?=[ակ])",
     'ignore_case': True,
     'replacement': [(move_first2end_samecase, 0.1)]}
)

patterns_dicts.append(
    {'pattern': fr'\b[{armchars_upper}](?![{armchars_upper}])',
     'ignore_case': False,
     'replacement': [(hy_lower, 0.05)]}
)
patterns_dicts.append(
    {'pattern': fr'\b[{armchars_upper}]+(?![{armchars_lower}])',
     'ignore_case': False,
     'replacement': [(hy_lower, 0.03), (cap_from_upper, 0.03)]}
)
patterns_dicts.append(
    {'pattern': fr'\b[{armchars_lower}]+',
     'ignore_case': False,
     'replacement': [(cap_from_lower, 0.02), (hy_upper, 0.001)]}
)
patterns_dicts.append(
    {'pattern': fr'(?<=(\n|\.|:) )[{armchars_upper}][{armchars_lower}]+(?![{armchars_upper}])',
     'ignore_case': False,
     'replacement': [(hy_lower, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr'(?<=(\n|\.|:) )[{armchars_upper}]+(?![{armchars_lower}])',
     'ignore_case': False,
     'replacement': [(hy_lower, 0.1), (cap_from_upper, 0.15)]}
)
patterns_dicts.append(
    {'pattern': fr'(?<=(\n|\.|:) )[{armchars_lower}]+',
     'ignore_case': False,
     'replacement': [(cap_from_lower, 0.1), (hy_upper, 0.01)]}
)
patterns_dicts.append(
    {'pattern': fr'(?<=(«|\(|\|) )[{armchars_upper}](?![{armchars_upper}])(.*?)(?=(»|\)|\|))',
     'ignore_case': False,
     'replacement': [(hy_lower, 0.1), (hy_upper,0.02)]}
)
patterns_dicts.append(
    {'pattern': fr'(?<=(«|\(|\|) )[{armchars_upper}]+(?![{armchars_lower}])(.*?)(?=(»|\)|\|))',
     'ignore_case': False,
     'replacement': [(hy_lower, 0.1), (cap_from_upper, 0.1)]}
)
patterns_dicts.append(
    {'pattern': fr'(?<=(«|\(|\|) )[{armchars_lower}]+(.*?)(?=(»|\)|\|))',
     'ignore_case': False,
     'replacement': [(cap_from_lower, 0.06), (hy_upper, 0.025), (cap_allwords, 0.02)]}
)
patterns_dicts.append(
    {'pattern': r'(?m)^.*[' + re.escape(armchars_lower) + r'].*$',
     'ignore_case': False,
     'replacement': [(hy_upper, 0.005), (cap_allwords, 0.01)]}
)
patterns_dicts.append(
    {'pattern': r'(?m)^(?=.*[' + re.escape(armchars_upper) + r'])(?!.*[' + re.escape(armchars_lower) + r']).+$\n',
     'ignore_case': False,
     'replacement': [(lower_text_random, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[ԵՈ])ւ",
     'ignore_case': False,
     'replacement': [(hy_upper, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r"(?<=[ԵՈ])Ւ",
     'ignore_case': False,
     'replacement': [(hy_lower, 0.1)]}
)

patterns_dicts.append(
    {'pattern': fr"(?<=[{armchars_lower}]) ",
     'ignore_case': True,
     'replacement': [(add_comma, 0.007), (add_midpoint, 0.001), (add_but, 0.004), (add_question, 0.0002), (add_emphasis, 0.0002), (add_exclamation, 0.00015), (add_endpoint, 0.002)]}
)
patterns_dicts.append(
    {'pattern': r"[.՞՛՜] ",
     'ignore_case': True,
     'replacement': [(' ', 0.12)]}
)

patterns_dicts.append(
    {'pattern': r"[՝`] ",
     'ignore_case': True,
     'replacement': [(' ', 0.13), (', ', 0.2)]}
)

patterns_dicts.append(
    {'pattern': r", ",
     'ignore_case': True,
     'replacement': [(' ', 0.12), ('՝ ', 0.12), (': ', 0.05), ('։ ', 0.05)]}
)

patterns_dicts.append(
    {'pattern': r"[:։] ",
     'ignore_case': True,
     'replacement': [(' ', 0.1), (', ', 0.05), ('. ', 0.05)]}
)

patterns_dicts.append(
    {'pattern': r"(?<=, )որ\b",
     'ignore_case': True,
     'replacement': [(write_vorpeszi, 0.05)]}
)

patterns_dicts.append(
    {'pattern': r"\bէ\b",
     'ignore_case': True,
     'replacement': [(add_a_samecase, 0.1)]}
)
patterns_dicts.append(
    {'pattern': r" (?=[էա]\b)",
     'ignore_case': True,
     'replacement': [(removematch, 0.08)]}
)
patterns_dicts.append(
    {'pattern': r" (?=(եմ|ես|եք|են|էի|էր))",
     'ignore_case': True,
     'replacement': [(removematch, 0.08)]}
)
patterns_dicts.append(
    {'pattern': r" (?=(ենք|էիր|էիք|էին))",
     'ignore_case': True,
     'replacement': [(removematch, 0.08)]}
)
patterns_dicts.append(
    {'pattern': r" (?=էինք)",
     'ignore_case': True,
     'replacement': [(removematch, 0.08)]}
)

patterns_dicts.append(
    {'pattern': rf'[{armchars_lower}]*՞[{armchars_lower}]*',
     'ignore_case': True,
     'replacement': [(rand_move_quest, 0.15)]}
)

patterns_dicts.append(
    {'pattern': r"՞",
     'ignore_case': True,
     'replacement': [(write_latquest, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r"՞[^:|\n]*[:\n]",
     'ignore_case': True,
     'replacement': [(write_latquest_before_end, 0.01), (write_latquest_after_end, 0.01), (write_latquest_at_end, 0.06)]}
)
patterns_dicts.append(
    {'pattern': r"՛",
     'ignore_case': True,
     'replacement': [(write_latemph, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r"՛[^:|\n]*[:\n]",
     'ignore_case': True,
     'replacement': [(write_latemph_at_end, 0.07)]}
)
patterns_dicts.append(
    {'pattern': fr"(?<=[{cons_l}])չ(?=[իո])",
     'ignore_case': True,
     'replacement': [(add_i_samecase, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r'\bվերած[եվո]\w* +(եմ\w*|ես\w*|են\w*|եք\w*|էր\w*|էի\w*|[էա])', # 'vերածվել' + aux_verb
     'ignore_case': True,
     'replacement': [(separate_veratsvel, 0.15)]}
)
patterns_dicts.append(
    {'pattern': r'\bվերած\w*\b(?! եմ\w*| ես\w*| են\w*| եք\w*| էր\w*| էի\w*| [էա]\b)', # 'vերածվել' (standalone)
     'ignore_case': True,
     'replacement': [(separate_veratsvel, 0.05)]}
)
patterns_dicts.append(
    {'pattern': r'(?<=\bառանց\b) +\b\w*[եա]լու\b', # 'vերածվել' (standalone)
     'ignore_case': True,
     'replacement': [(add_u_samecase_end, 0.15)]}
)

with open('replacements.json', 'r', encoding='utf-8') as f:
    patterns_dicts += json.load(f)


# probas = []
# for patterns_dict in patterns_dicts:
#     for rep in patterns_dict['replacement']:
#         probas.append(rep[-1])
with open("num_patterns.json", "r") as json_file:
    mistake_dict = json.load(json_file)
min_freq = 9
mistake_scalers = [1 + np.log(freq/min_freq) for key, freq in mistake_dict.items()]

min_freq_p = 0.1
scaled_max_p = 0.25/min_freq_p

for i, patterns_dict in enumerate(patterns_dicts):
    updated_replacement = []
    for rep in patterns_dict['replacement']:
        p = scaled_max_p * rep[-1] / mistake_scalers[i]
        new_rep = (rep[0], p)
        updated_replacement.append(new_rep)
    patterns_dict['replacement'] = updated_replacement