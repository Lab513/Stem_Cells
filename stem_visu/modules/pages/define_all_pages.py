#!/usr/bin/env python
# encoding: utf-8
"""
define_all_pages.py
Build the variables used by jinja needed for the views
"""
import re
import os
import os.path as op
opd, opb, opj = op.dirname, op.basename, op.join

Interface_subtitle = ""


def scan_processed(path):
    '''
    Find the list of the processed folders
    '''
    list_dir = []
    for item in os.listdir(path):
        print(item)
        if item not in ['.DS_Store']:
            list_dir.append(item)
    return list_dir


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    take number in the name
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


class define_page(object):
    '''
    General template
    '''
    def __init__(self):
        self.body = {}
        self.header = {}
        self.footer = {}
        ### Body
        self.body['main_title'] = ""
        self.body['subtitle'] = ""
        self.body['explanations'] = ""
        ### Header
        self.header['main_title'] = ""
        self.header['presentation_interface'] = ""
        self.header['background'] = op.join('')

        ### Footer
        self.footer['background'] = op.join('static/img/black.jpg')
        self.footer['copyright'] = "no Copyright, version 0.0.0"


class define_main_page(define_page):
    def __init__(self, platf):
        '''
        Page (index_folder.html) for entering
         the parameters and launching the processings.
        '''
        define_page.__init__(self)
        self.platf = platf
