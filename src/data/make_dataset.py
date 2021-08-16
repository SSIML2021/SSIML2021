import os
import re
from pathlib import Path

import pandas as pd
from langdetect import detect
from nltk.tokenize import word_tokenize
from termcolor import colored


DOCUMENTS_FILEPATH = Path('data') / 'documents'
ANNOTATIONS_FILEPATH = Path('data') / 'annotations'


def read_data_file(file_name):
    return pd.read_csv(file_name, encoding="latin1")


def fix_speech_identifier(speech_identifier):
    speech_identifier = re.sub("Simor 2010-05-25", "Simor 2010-05-26", speech_identifier)
    speech_identifier = re.sub("^Mario ", "Draghi ", speech_identifier)
    speech_identifier = re.sub("^PM ", "Cameron ", speech_identifier)
    speech_identifier = re.sub("^Thorning ", "Thorning-Schmidt ", speech_identifier)
    speech_identifier = re.sub("Remarks 2009-12-11", "Honohan 2009-12-11", speech_identifier)
    speech_identifier = re.sub("This 2013-02-11", "Cameron 2013-02-11", speech_identifier)
    speech_identifier = re.sub("Mervyn ", "King ", speech_identifier)
    speech_identifier = re.sub("Patrick 2013-03-19", "Honohan 2013-03-19", speech_identifier)
    speech_identifier = re.sub("Statement 2014-12-18", "Kenny 2014-12-19", speech_identifier)
    speech_identifier = re.sub("Speech 2012-06-29", "Cameron 2012-06-29", speech_identifier)
    speech_identifier = re.sub("The 2012-01-30", "Cameron 2012-01-30", speech_identifier)
    speech_identifier = re.sub("Orban ", "Orbán ", speech_identifier)
    speech_identifier = re.sub("The 2014-10-24", "Cameron 2014-10-24", speech_identifier)
    speech_identifier = re.sub("Speech 2013-03-07", "Kenny 2013-07-03", speech_identifier)
    speech_identifier = re.sub("David 2014-11-10", "Cameron 2014-11-10", speech_identifier)
    speech_identifier = re.sub("Statement 2012-07-04", "Kenny 2012-07-04", speech_identifier)
    speech_identifier = re.sub("Schröder", "Schroeder", speech_identifier)
    speech_identifier = re.sub("Schroeder 1998-12-14", "Schroeder 1999-12-14", speech_identifier)
    speech_identifier = re.sub("Schroeder 2001-10-26", "Schroeder 2001-10-16", speech_identifier)
    speech_identifier = re.sub("Hollande 2015-05-19", "Hollande 2015-03-19", speech_identifier)
    speech_identifier = re.sub("Fernandez 2009-11-23", "Fernández Ordóñez  2009-11-23", speech_identifier)
    return speech_identifier


def get_speech_id(file_name, speeches):
    try:
        file_name_parts = file_name.split()
        date = file_name_parts[0]
        speaker = list(file_name_parts[1].split("_")[0])
        speaker[0] = speaker[0].upper()
        speaker = "".join(speaker)
        speech_identifier = f"{speaker} {date}"
        speech_identifier = fix_speech_identifier(speech_identifier)
        return int(speeches[speeches["Speech_Identifier"] == speech_identifier]["Speech_ID"])
    except:
        return None


def get_paragraph_ids(speech_id, speech_contents):
    paragraph_ids = {}
    try:
        for i, row in speech_contents[speech_contents["Speech_ID"] == speech_id].iterrows():
            paragraph_ids[row["Speech_Content_ID"]] = row["Speech_Content_Title"]
    except:
        pass
    return paragraph_ids


def check_paragraphs(speech_id, paragraph_ids, map_contents, file_name):
    paragraph_values = {}
    for i, row in map_contents[map_contents["Content_Speech_ID"] == speech_id].iterrows():
        if row["Content_Source_ID"] not in paragraph_ids:
            print(colored(f'warning: unknown paragraph id {row["Content_Source_ID"]} for document {speech_id}; file name: {file_name}', "red"))
        else:
            paragraph_values[f'{speech_id} {paragraph_ids[row["Content_Source_ID"]]}'] = True
    return paragraph_values


def read_paragraphs(file_name):
    paragraph_list = []
    data_file = open(file_name, "r", encoding="latin1")
    for line in data_file:
        paragraph_list.append(line.strip())
    data_file.close()
    return paragraph_list


def select_paragraphs(paragraph_list, paragraph_values, speech_id):
    paragraph_texts = {}
    use_paragraph = False
    for paragraph in paragraph_list:
        tokens = paragraph.strip().split()
        if len(tokens) > 0 and re.search(r'^\d+-\d+:*$', tokens[0]):
            use_paragraph = True
            key = re.sub(":", "", tokens[0])
            key = f"{speech_id} {key}"
            tokens.pop(0)
            if len(tokens) > 0 and tokens[0] == ":":
                tokens.pop(0)
        if len(tokens) > 0 and use_paragraph:
            paragraph_texts[key] = " ".join(word_tokenize(" ".join(tokens))).lower()
            if key not in paragraph_values:
                paragraph_values[key] = False
            use_paragraph = False
    return paragraph_texts


def guess_language(paragraph_texts):
    text = " ".join(paragraph_texts.values())
    try:
        return detect(text)
    except:
        return "unk"


def make_dataset(speeches, speech_contents, map_contents):
    paragraph_texts_all = {}
    paragraph_values_all = {}
    nbr_of_files = 0
    nbr_of_skipped = 0
    for file_name in DOCUMENTS_FILEPATH.iterdir():
        speech_id = get_speech_id(str(file_name), speeches)
        if speech_id is None:
            print(f"skipping file {file_name}")
            nbr_of_skipped += 1
        else:
            paragraph_ids = get_paragraph_ids(speech_id, speech_contents)
            paragraph_values = check_paragraphs(speech_id, paragraph_ids, map_contents, file_name)
            paragraph_list = read_paragraphs(file_name)
            paragraph_texts = select_paragraphs(paragraph_list, paragraph_values, speech_id)
            language = guess_language(paragraph_texts)
            if language == "en":
                if len(paragraph_texts) != len(paragraph_values):
                    print(colored(f"warning: mismatch meta data ({len(paragraph_values)}) vs file ({len(paragraph_texts)}) for file {file_name}", "red"))
                paragraph_texts_all.update(paragraph_texts)
                paragraph_values_all.update(paragraph_values)
                nbr_of_files += 1
            else:
                print(f"skipping file in language {language}: {file_name}")
                nbr_of_skipped += 1
    print(f"read {nbr_of_files} files with {len(paragraph_values_all)} paragraphs; skipped {nbr_of_skipped} file", end="")
    if nbr_of_skipped != 1:
        print("s")
    else:
        print("")
    return paragraph_texts_all, paragraph_values_all


def read_annotations():
    map_contents = read_data_file(ANNOTATIONS_FILEPATH / "Map_Contents-20200726.csv")
    speech_contents = read_data_file(ANNOTATIONS_FILEPATH / "Speech_Contents-20210520.txt")
    speeches = read_data_file(ANNOTATIONS_FILEPATH / "Speeches-20210520.txt")
    return map_contents, speech_contents, speeches
