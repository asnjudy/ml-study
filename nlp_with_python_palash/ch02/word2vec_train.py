import random
import collections
import math
import os
import zipfile
import time
import re
import numpy as np
import tensorflow as tf

from matplotlib import pyplot
from six.moves import range
from six.moves.urllib.request import urlretrieve

with open('text8') as ft_:
    full_text = ft_.read()


def text_processing(ft8_text):
    """
    :type ft8_text: str
    :param ft8_text:
    :return:
    """
    ft8_text = ft8_text.lower()
    ft8_text = ft8_text.replace('.', '<period>')
    ft8_text = ft8_text.replace(',', '<comma>')
    ft8_text = ft8_text.replace('"', '<quotation>')
    ft8_text = ft8_text.replace(';', '<semicolon>')
    ft8_text = ft8_text.replace('!', '<exclamation>')
    ft8_text = ft8_text.replace('?', '<question>')
    ft8_text = ft8_text.replace('(', '<paren_l>')
    ft8_text = ft8_text.replace(')', '<paren_r>')
    ft8_text = ft8_text.replace('--', '<hyphen>')
    ft8_text = ft8_text.replace(':', '<colon>')
    ft8_text_tokens = ft8_text.split()  # 默认以空格分隔
    return ft8_text_tokens


ft_tokens = text_processing(full_text)

print()