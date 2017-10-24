from __future__ import absolute_import

import datetime
import inspect
import numpy as np
import os
import pdb
import tensorflow as tf
import datetime

from terminaltables import SingleTable, AsciiTable, GithubFlavoredMarkdownTable

ROUND_PREFIX = "round"
ROUND_SUFFIX = "-"

NOTES_BY_CALLER = {}

def noteworthy_log(**kwargs):
    current_time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_frame = inspect.currentframe()
    calling_frame = inspect.getouterframes(current_frame, 2)
    caller_name = calframe[1][3]

    if not (caller_name in NOTES_BY_CALLER):
        NOTES_BY_CALLER[caller_name] = []

    NOTES_BY_CALLER[caller_name].append((current_time_string, dict(kwargs)))

def create_folder_if_not_exists(path, if_not_message):
    if not os.path.exists(path):
        os.mkdir(path)
        print if_not_message.format(path)

def create_latest_symlink(path, latest_relative_path):
    if os.path.islink(latest_relative_path):
        os.unlink(latest_relative_path)

    os.symlink(path, latest_relative_path)

def ensure_shape_3d(shape):
    if len(shape) < 2 or len(shape) > 3:
        raise ValueError("expected at worst, a 2D shape")

    if len(shape) == 3:
        return shape

    return shape + (3,)

def evenly_divided_by(n, m):
    return (n % m) == 0

def shape_as_tuple(sh):
    return tuple(map(int, sh))

def shape_as_string(sh):
    if sh is None:
        return "None"

    return "x".join(map(lambda el: str(el), sh))

def managed_or_debug_session(config, scaffold, hooks, debug=False):
    if debug:
        print "debug mode: disabling any monitoring."
        session = tf.Session(config=config)
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        tf.train.start_queue_runners(session)

        return session

    return tf.train.SingularMonitoredSession(hooks, scaffold=scaffold, config=config)

def value_or_default(data, key, default, parse_function=None, required=False):
    if parse_function is None:
        parse_function = lambda x: x
        
    if key in data:
        return parse_function(data[key])

    if (default is None) and required:
        raise ValueError("Failed to find \"{0}\" and no default was given".format(key))
    
    return default

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def debug_description(description_kv, markdown=True):
    # always sort of the dictionary for consistency.
    sorted_description_kv = sorted(description_kv.items(), key=lambda kv: kv[0])
    description_rows = [["name", "value"]] + map(lambda kv: [str(kv[0]), str(kv[1])], sorted_description_kv)

    if markdown:
        description_table = GithubFlavoredMarkdownTable(description_rows)
    else:
        description_table = AsciiTable(description_rows)

        description_table.inner_heading_row_border = False
        description_table.outer_border = False
    
    return str(description_table.table)

def render_summary(summary, title=None, outer_border=True):
    # we should split long lines here at some point.
    table = AsciiTable(summary, title=title)
    table.inner_heading_row_border = False
    table.outer_border = outer_border
    
    return str(table.table)

def float_equals(x, y, epsilon=0.00000001):
    return abs(x-y) < epsilon

def increment_round_number(name):
    if name.startswith(ROUND_PREFIX):
        end_index = name.index(ROUND_SUFFIX)
        current_round = int(name[len(ROUND_PREFIX):end_index])

        return ROUND_PREFIX + str(current_round + 1) + ROUND_SUFFIX + name[end_index + 1:]

    return ROUND_PREFIX + "1"  + ROUND_SUFFIX + name

def replace_round_number(name, round_number):
    if name.startswith(ROUND_PREFIX):
        end_index = name.index(ROUND_SUFFIX)
        current_round = int(name[len(ROUND_PREFIX):end_index])

        return ROUND_PREFIX + str(round_number) + ROUND_SUFFIX + name[end_index + 1:]

    return ROUND_PREFIX + str(round_number) + ROUND_SUFFIX + name

def get_round_number(name):
    round_number = 0

    if name.startswith(ROUND_PREFIX):
        round_number = int(name[len(ROUND_PREFIX):name.index(ROUND_SUFFIX)])

    return round_number

def get_round_label_string(name_label):
    round_number = 0
    name, label = name_label

    round_number = get_round_number(name)

    return "RD{0}_C{1}".format(round_number, label)

def random_image_normal(width, height, channels=3):
    return np.random.normal(0, 0.3, size=(1, height, width, channels))

def random_image(width, height, block_size=8):
    # temp = np.random.uniform(0.0,1.0)
    temp = 1.0
    if temp > 0.5:
        res = np.zeros((1, height, width, 3))
        bal_ = np.random.uniform(-0.5, 0.5)
        for i in range(0, height / block_size):
            for j in range(0, width / block_size):
                mean_ = np.random.uniform(-0.8, 0.8)
                var_ = np.random.uniform(0.1, 0.8)
                res[0, block_size*i:block_size*i+block_size, block_size*j:block_size*j+block_size, :] = np.random.normal(mean_, var_, size=(block_size, block_size, 3))
        return (res+bal_).astype('float32')
    else:
        res = np.zeros((1, height, width, 3))
        for i in range(0, block_size, 1):
            for j in range(0, block_size, 1):
                mean_ = np.random.uniform(-0.8, 0.8)
                var_ = np.random.uniform(0.1, 0.7)
                res[0, block_size*i:block_size*i+block_size, block_size*j:block_size*j+block_size, 0] = np.random.normal(mean_, var_)
                mean_ = np.random.uniform(-0.8, 0.8)
                var_ = np.random.uniform(0.1, 0.7)
                res[0, block_size*i:block_size*i+block_size, block_size*j:block_size*j+block_size, 1] = np.random.normal(mean_, var_)
                mean_ = np.random.uniform(-0.8, 0.8)
                var_ = np.random.uniform(0.1, 0.7)
                res[0, block_size*i:block_size*i+block_size, block_size*j:block_size*j+block_size, 2] = np.random.normal(mean_, var_)
        return res.astype('float32')


def parse_date_string(date_string):
    return datetime.datetime.strptime(date_string, "%m%d%Y_%H%M%S")    
    
def format_date_string(date):    
    return date.strftime("%B %d %Y at %I:%M:%S %p")

def parse_date_string(date_string):
    return datetime.datetime.strptime(date_string, "%m%d%Y_%H%M%S")

def format_date_string(date_string):
    return parse_date_string(date_string).strftime("%B %d %Y at %I:%M:%S %p")

def prompt_for_choice(title, choices):
    number_choices = len(choices)

    print title
    for choice_index in range(number_choices):
        print "[{0}] {1}".format(choice_index, choices[choice_index])

    choice_index = None
    while True:
        choice_verbatim = raw_input("Select package: ")

        try:
            choice_index = int(choice_verbatim)
        except:
            print "Invalid!"
            continue

        if choice_index < 0 or choice_index >= number_choices:
            print "Choice is out of range..."
            continue
        
        return choice_index

def prompt_for_run(model_path, detail=None):
    run_names = filter(lambda d: not d.endswith(".zip"), os.listdir(model_path))
    run_names = list(sorted(run_names, key=parse_date_string))
    run_titles = map(format_date_string, run_names)

    prompt_name = "Select a run"
    if detail:
        prompt_name = prompt_name + " ({0})".format(detail)
        
    chosen_experiment_index = prompt_for_choice(prompt_name, run_titles)

    return run_names[chosen_experiment_index]
