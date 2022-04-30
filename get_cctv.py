import os
import argparse
import datetime
from enum import Enum

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

import requests
from bs4 import BeautifulSoup as bs
from html.parser import HTMLParser


MAIN_URL = 'https://cn.govopendata.com/xinwenlianbo/'
SUFFIX = '_text.csv'


class ReturnCode(Enum):
    WRITE_SUCCESS = 0
    WRITE_FAILURE = 1
    WRITE_ALREADY_EXISTS = 2
    WRITE_EMPTY = 3
    WRITE_NUM_MISMATCH = 4
    WRITE_NO_URL_OR_HTML = 5


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--location', '-l', default='../data',
                        help='directory to save file')

    parser.add_argument('--history', '-his', default=False, action='store_true',
                        help='write news for today and all prior available')

    parser.add_argument('--year', '-y', type=int, default=None,
                        help='write news for specific year')

    parser.add_argument('--month', '-m', type=int, default=None,
                        help='write news for specific month')

    parser.add_argument('--day', '-d', type=int, default=None,
                        help='write news for specific day')

    args = parser.parse_args()
    return args


def get_date(delta=0, **kwargs):
    try:
        date = datetime.date(**kwargs)
    except:
        date = datetime.date.today()
    date += datetime.timedelta(days=delta)
    return date


def get_url(date):
    y, m, d = date.year, date.month, date.day
    url = MAIN_URL + '%4d%02d%02d' % (y, m, d)
    return url


def get_html(url):
    req = requests.get(url)
    html = bs(req.text, 'html.parser')
    return html


def get_attr(html, *args, **kwargs):
    return html.find_all(*args, **kwargs)


def get_titles(html):
    return get_attr(html, 'h2')


def get_passages(html):
    return get_attr(html, 'p')


def write_1_day(location, delta=0, **kwargs):
    date = get_date(delta=delta, **kwargs)

    os.makedirs(location, exist_ok=True)
    filename = os.path.join(location, str(date) + SUFFIX)
    if os.path.isfile(filename):
        return filename, ReturnCode.WRITE_ALREADY_EXISTS

    try:
        url = get_url(date)
        html = get_html(url)
    except:
        return filename, ReturnCode.WRITE_NO_URL_OR_HTML

    try:
        titles = get_titles(html)
        passages = get_passages(html)
        assert len(titles) > 0, ReturnCode.WRITE_EMPTY
        assert len(titles) == len(passages), ReturnCode.WRITE_NUM_MISMATCH
    except:
        return filename, ReturnCode.WRITE_EMPTY

    try:
        file = open(filename, 'w', encoding="utf-8")
        lines = ['id\ttitle\tpassage\n']

        for t, p in zip(titles, passages):
            i = t.contents[0]['id']
            t = t.text.replace('\t', ' ').replace('\n', ' ')
            p = p.text.replace('\t', ' ').replace('\n', ' ')
            lines.append('\t'.join([i, t, p]) + '\n')

        file.writelines(lines)
    except Exception as e:
        file.close()
        os.remove(filename)
        print(e)
        return filename, ReturnCode.WRITE_FAILURE

    file.close()
    return filename, ReturnCode.WRITE_SUCCESS


def write_history(location):
    end = 1 + (datetime.date.today() - datetime.date(2007, 1, 1)).days
    last_status = 0
    
    for delta in range(end):
        filename, status = write_1_day(location, delta=-delta)
        print(filename, status)
        last_status = last_status + 1 if status != ReturnCode.WRITE_SUCCESS else 0 # and status != ReturnCode.WRITE_ALREADY_EXISTS else 0
        if last_status == 4:
            break
        
def proc(d, *args, **kwargs):
    f, s = write_1_day(*args, **kwargs)
    d[f] = s
    return 1

def write_history_multithread(location):
    today = datetime.date.today()
    end = 1 + (today - datetime.date(2007, 1, 1)).days

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    futures = []
    d = dict()

    for delta in range(end):
        futures.append(executor.submit(partial(proc, d, location, -delta)))
    results = [future.result() for future in tqdm(futures)]
    with open('dl_%s.log' % today, 'w') as file:
        for k, v in d.items():
            file.write(k + ' ' + v + '\n')

def main():
    args = parse_args()
    if args.history:
        write_history_multithread(args.location)
    else:
        filename, status = write_1_day(
            args.location, year=args.year, month=args.month, day=args.day)
        print(filename, status)


if __name__ == '__main__':
    main()
