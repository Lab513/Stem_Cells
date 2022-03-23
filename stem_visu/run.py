#!/usr/bin/env python
# encoding: utf-8

"""

Visualiser for stem_cells predictions

python -m stem_visu.run

"""
from __future__ import print_function, division, absolute_import

# Flask imports
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from stem_visu.modules.pages.define_all_pages import *
from stem_visu.modules.util_interf import *

import os
import glob
import yaml
import json
import shutil as sh
import subprocess
from colorama import Fore, Style                # Color in the Terminal
import webbrowser
import threading

op = os.path
opd, opb, opj = op.dirname, op.basename, op.join

import socket

platf = find_platform()

if platf == 'win':
    import gevent as server
    from gevent import monkey
    monkey.patch_all()
else:
    import eventlet as server
    server.monkey_patch()

Debug = True                                                # Debug Flask

app = Flask(__name__)
app.config['UPLOADED_PATH'] = opj(os.getcwd(), 'stem_visu', 'upload')
print("######### app.config['UPLOADED_PATH'] "
      " is {app.config['UPLOADED_PATH']} !!!")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'F34TF$($e34D'
socketio = SocketIO(app)


@socketio.on('connect')
def test_connect():
    '''
    Websocket connection
    '''
    send_wells_list()
    send_nb_pics()
    send_nb_cells_max()
    send_infos()
    emit('response', 'Connected')
    server.sleep(0.05)


@app.route('/', methods=['GET', 'POST'])
def main_page(debug=1):
    '''
    '''
    platf = find_platform()
    print(f'platf is { platf }')
    dmp = define_main_page(platf)
    return render_template('index_folder.html', **dmp.__dict__)


def send_nb_pics():
    '''
    Retrieve and send the number of pictures
    '''
    lwells = make_list_wells()
    addr_res = f'stem_visu/static/results/pred_{lwells[0]}'
    ll = glob.glob(opj(addr_res, 'img*.png'))
    nb_pics = int(len(ll)/3)              # not taking superp into account...
    print(f'nb of pics is { nb_pics }')
    emit('nb_pics', str(nb_pics))


def make_list_wells():
    '''
    List of the processed wells
    '''
    root_res = 'stem_visu/static/results'
    ll = glob.glob(opj(root_res, 'pred_*'))
    ll_sorted = sorted(ll, key= lambda elem : int(re.findall('(\\d+)', elem)[0]))
    lwells = []
    for l in ll_sorted:
        print(l)
        print(opb(l))
        print(opb(l).split('_'))
        well = opb(l).split('_')[1]
        lwells += [well]

    return lwells


def send_wells_list():
    '''
    Retrieve the wells processed and send the list to the interface.
    The unprocessed wells are colored in black
    '''
    lwells = make_list_wells()
    print(f'lwells is { lwells }')
    emit('lwells', json.dumps(lwells))


def send_nb_cells_max(debug=[0]):
    '''
    Send the number of cells_max
     for each well to the interface
     for determining the color of the wells
    '''
    lwells = make_list_wells()
    dic_nb_cells_max = {}
    for w in lwells:
        addr_nbmax = f'stem_visu/static/results/pred_{w}/nb_cells_max.yaml'
        nbmax = 0
        try:
            with open(addr_nbmax) as f_r:
                nbmax = yaml.load(f_r, Loader=yaml.FullLoader)
        except:
            print('Probably no nb_cells_max.yaml file')
        dic_nb_cells_max[w] = nbmax
    if 0 in debug:
        print(f'dic_nb_cells_max is { dic_nb_cells_max }')
    emit('dic_nb_cells_max', json.dumps(dic_nb_cells_max))


def send_infos():
    '''
    Infos about the experiment
    '''
    exp_infos = f'stem_visu/static/results/proc_infos.yaml'
    with open(exp_infos) as f_r:
        infos = yaml.load(f_r, Loader=yaml.FullLoader)
        print(f'infos are {infos}')
    emit('exp_infos', json.dumps(infos))


@socketio.on('mess')
def receiving_mess(mess):
    '''
    Receiving a message
    '''
    print(f"mess is { mess }")
    emit('refresh', "")


def shutdown_server():
    '''
    Quit the application
    called by method shutdown() (hereunder)
    '''
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown')
def shutdown():
    '''
    Shutting down the server.
    '''
    shutdown_server()

    return 'Server shutting down...'


def message_at_beginning(host, port):
    '''
    '''
    print(Fore.YELLOW + f"""
    ***************************************************************
    Launching the stem_visu server program !!!

    address: { host }:{ port }

    """)


def find_chrome_path(platf):
    '''
    '''
    # MacOS
    if platf == 'mac':
        chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    # Linux
    elif platf == 'lin':
        chrome_path = '/usr/bin/google-chrome %s'
    else:
        chrome_path = False
    return chrome_path


def launch_browser(port, host, platf):
    '''
    Launch Chrome navigator
    '''
    chrome_path = find_chrome_path(platf)
    url = f"http://{ host }:{port}"
    if platf != 'win':
        b = webbrowser.get(chrome_path)
        # open a page in the browser.
        threading.Timer(1.25, lambda: b.open_new(url)).start()
    else:
        try:
            print('using first path')
            subprocess.Popen(f'"C:\Program Files\Google\Chrome\Application\chrome.exe" {url}')
        except:
            print('using second path')
            subprocess.Popen(f'"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" { url }')


if __name__ == '__main__':
    init(app.config)             # clean last processings and upload folders

    ip = socket.gethostbyname(socket.gethostname())
    port = 5988
    host = ip
    print("host is ", host)
    launch_browser(port, host, platf)
    message_at_beginning(host, port)
    print(Style.RESET_ALL)
    socketio.run(app, port=port, host=host)
