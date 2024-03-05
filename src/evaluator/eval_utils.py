from typing import Dict, List
from nltk import sent_tokenize
import re

def extract_citations(text) -> List[int]:
    """
    Extract all citations from text and merge them into a list. The paratheses can be 【】or [], and the citations can be 1,2,3 or 1，2，3 or 1-3
    Example:
        input："这是第一句话。【1，2】这是第二句话[1][3]。这是第三句话[1-4]。这是【最后】一句话。[1-5,7]"
        otuput：[1, 2, 3, 4, 5, 7]
    """
    def _replace_delimiter(text):
        return text.replace('，', ',').replace('、', ',')
    pattern = r'【([\d,，、 -]+)】|\[([\d,，、 -]+)\]' # 【dilimited numbers】or [dilimited numbers]
    matches = re.findall(pattern, text)
    matches = [match[0] if match[0] else match[1] for match in matches]
    citations = []
    for match in matches:
        if re.findall(r'\d-\d', match): # string such as 1,3-5,7, should be mapped to [1,3,4,5,7]
            start, end = match.split('-')[:2]
            start = [int(cite.strip()) for cite in _replace_delimiter(start).split(',') if cite.strip().isdecimal()]
            end = [int(cite.strip()) for cite in _replace_delimiter(end).split(',') if cite.strip().isdecimal()]
            if not start or not end:
                continue
            if len(start) > 1: # [1,3]
                citations.extend(start[:-1])
            start = start[-1]
            if len(end) > 1: # [5,7]
                citations.extend(end[1:])
            end = end[0]
            citations.extend(list(range(int(start), int(end)+1))) # range(3,6)
        else:
            citations.extend([int(cite.strip()) for cite in _replace_delimiter(match).split(',') if cite.lstrip('-').strip().isdecimal()])
    return sorted(list(set(citations)))


def remove_citations(text) -> str:
    """
    Remove all citations from text.
    Example:
        input："这是第一句话。【1，2】这是第二句话[1][3]。这是第三句话[1-4]。这是【最后】一句话。[1-5,7]"
        otuput：'这是第一句话。这是第二句话。这是第三句话。这是【最后】一句话。'
    """
    pattern = r'【([\d, ，、-]+)】|\[([\d, ，、-]+)\]' # 【citation numbers】or [citation numbers]
    return re.sub(pattern, '', text)


def longest_common_substring(str1, str2) -> str:
    m, n = len(str1), len(str2)
    dp, max_len, end_idx = [[0] * (n + 1) for _ in range(m + 1)], 0, 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if str1[i - 1] == str2[j - 1] else 0
            if dp[i][j] > max_len:
                max_len, end_idx = dp[i][j], i - 1
    return str1[end_idx - max_len + 1:end_idx + 1]


def parse_text_with_citations(text: str, lan='chinese') -> List[Dict]:
    """
    Input: a piece of text embedded with in-line citations
    Return: a list of dict mapping each `sentence` to its `citations`; if `citations` of a sentence is empty, 
            will look for the nearest citations behind the sentence.
    Example:
        input："这是第一句话。【1，2】这是第二句话[1][3]。这是第三句话。这是【最后】一句话。[1-5,7]"
        output:[{'sentence': '这是第一句话。', 'citations': [1, 2], 'nearest_citations': [1, 2]'},
                {'sentence': '这是第二句话。', 'citations': [1, 3], 'nearest_citations': [1, 3]'},
                {'sentence': '这是第三句话。', 'citations': [], 'nearest_citations': [1, 2, 3, 4, 5, 7]'},
                {'sentence': '这是【最后】一句话。', 'citations': [1, 2, 3, 4, 5, 7], 'nearest_citations': [1, 2, 3, 4, 5, 7]}]
    """
    if not text:
        return []
    # segment sentences
    if lan != 'chinese':
        sentences = sent_tokenize(text, language=lan)
    else:
        segments = [sen for sen in re.split('(?<=[。；！？;!?\n])', remove_citations(text))] 
        sentences = []
        for seg in segments:
            if len(seg.strip()) > 1: # sentence expected
                sentences.append(seg)
            elif len(sentences) > 0: # if seg is a punctuation, append it to the last sentence
                sentences[-1] += seg

    sen_offsets = []
    for sen in sentences:
        if sen[:-1] in text: # sentence without punctuation, most of the time
            sen_offsets.append(text.find(sen[:-1]))
        else: # if citations are embedded in the middle of a sentence, causing the sentences stripped of citations not to match the original sentences
            lcs = longest_common_substring(sen, text)
            sen_offsets.append(text.find(lcs))

    pattern = r'【([\d,，、 -]+)】|\[([\d,，、 -]+)\]' # 【citation numbers】or [citation numbers]
    citation_offsets = [(match.start(), match.end()) for match in re.finditer(pattern, text)]
    text_parsing = []
    for i in range(len(sentences)):
        item = {"sentence": sentences[i], "citations": []}
        sen_idx = sen_offsets[i]
        next_sen_idx = sen_offsets[i+1] if i < len(sentences)-1 else len(text)+1
        for start, end in citation_offsets: # match each citation to the nearest sentence before it
            if sen_idx < start and end <= next_sen_idx:
                item["citations"].extend(extract_citations(text[start:end]))
        item["citations"] = sorted(list(set(item["citations"])))
        text_parsing.append(item)

    # look for nearest citations
    for i in range(len(text_parsing)):
        text_parsing[i]["nearest_citations"] = text_parsing[i]["citations"]
        if i < len(text_parsing)-1 and not text_parsing[i]["citations"]:
            for j in range(i+1, len(text_parsing)):
                if text_parsing[j]["citations"]:
                    text_parsing[i]["nearest_citations"] = text_parsing[j]["citations"]
                    break

    return text_parsing


def build_text_from_parsing(text_parsing: List[Dict], citation_key='citations') -> str:
    """
    Convert the parsed text back to the original text with citations embedded.
    """
    embedded_text = ""
    for item in text_parsing:
        embedded_text += item["sentence"]
        if item[citation_key]==[] or ('citation_mask' in item.keys() and item['citation_mask']==0):
            continue
        else:
            citations = ''.join([f'[{cite}]' for cite in item[citation_key]])

            if embedded_text[-1] == '\n':
                text, lbreak = embedded_text[:-1], '\n'
            else:
                text, lbreak = embedded_text, ''

            if text[-1] in '。；！？：;!?:':
                embedded_text = text[:-1] + citations + text[-1] + lbreak
            else:
                embedded_text = text + citations + lbreak
    return embedded_text

