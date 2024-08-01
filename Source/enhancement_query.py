# -*- coding: utf-8 -*-
#!pip install python-docx

import json
import re
import pandas as pd

class EnhancementQuery():
  def __init__(self,file_name_terms, file_name_abbreviations):
    with open (file_name_terms, "r") as file:  self.terms_definitions = json.load(file)
    with open (file_name_abbreviations, "r") as file:  self.abbreviations_definitions = json.load(file)

  def preprocess(self, sentence,lowercase=True):
    """Converts text and optionally converts to lowercase. Removes punctuation."""
    if lowercase:
        sentence = sentence.lower()
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for char in punctuations:
        sentence = sentence.replace(char, '')
    return sentence

  def find_and_filter_terms(self,terms_dict, sentence):

    """Finds terms in the given sentence, case-insensitively, and filters out shorter overlapping terms."""
    ## Preprocess the text input
    lowercase_sentence = self.preprocess(sentence, lowercase=True)

    # Find all terms
    matched_terms = {term: terms_dict[term] for term in terms_dict if self.preprocess(term) in lowercase_sentence}

    # Filter out terms that are subsets of longer terms
    final_terms = {}
    for term in matched_terms:
        if not any(term in other and term != other for other in matched_terms):
            final_terms[term] = matched_terms[term]

    return final_terms

  def find_and_filter_abbreviations(self, abbreviations_dict, sentence):
    """Finds abbreviations in the given sentence, case-sensitively, and filters out shorter overlapping abbreviations."""
    processed_sentence = self.preprocess(sentence, lowercase=False)
    words = processed_sentence.split()

    # Filter all abbreviations that make match in a dictionary
    matched_abbreviations = {word: abbreviations_dict[word] for word in words if word in abbreviations_dict}

    final_abbreviations = {}
    sorted_abbrs = sorted(matched_abbreviations, key=len, reverse=True)

    for abbr in sorted_abbrs:
        if not any(abbr in other and abbr != other for other in sorted_abbrs):
            final_abbreviations[abbr] = matched_abbreviations[abbr]

    return final_abbreviations

  def find_terms_and_abbreviations_in_sentence(self,terms_dict, abbreviations_dict, sentence):
    """Finds and filters terms or abbreviations in the given sentence.
       Abbreviations are matched case-sensitively, terms case-insensitively, and longer terms are prioritized."""
    processed_sentence = self.preprocess(sentence, lowercase=False)  # Preserve case for abbreviations

    matched_abbreviations = {abbr: abbreviations_dict[abbr] for abbr in abbreviations_dict if abbr in processed_sentence}

    # Find and filter terms
    matched_terms = self.find_and_filter_terms(terms_dict, sentence)

    # Format matched terms and abbreviations for output
    formatted_terms = [f"{term}: {definition}" for term, definition in matched_terms.items()]
    formatted_abbreviations = [f"{abbr}: {definition}" for abbr, definition in matched_abbreviations.items()]

    return formatted_terms, formatted_abbreviations

  def define_TA_question(self, sentence):

    sentence = sentence[0:sentence.find("[")]

    formatted_terms, formatted_abbreviations = self.find_terms_and_abbreviations_in_sentence(self.terms_definitions,
                                                                                        self.abbreviations_definitions,
                                                                                       sentence)
    if len(formatted_terms) !=0:
      terms= '\n'.join(formatted_terms)
    else:
      terms = None

    if len(formatted_abbreviations)!=0:

      abbreviations= '\n'.join(formatted_abbreviations)
    else:
      abbreviations=None

    return [terms,abbreviations]



