# -*- coding: utf-8 -*-

"""
Created on 08/22/2021
mail_log.
@author: AnonymousUser314156
"""

from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
import smtplib

import requests
import random
import re


def validateEmail(email):

    if len(email) > 7:
        if re.match("^.+\\@(\\[?)[a-zA-Z0-9\\-\\.]+\\.([a-zA-Z]{2,3}|[0-9]{1,3})(\\]?)$", email) != None:
            return 1
    return 0


def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def get_sentence():
    url = "http://open.iciba.com/dsapi/"
    r = requests.get(url)
    # print(r.json())
    contents = r.json()['content']
    note = r.json()['note']
    translation = r.json()['translation']
    print(contents, note)
    return contents + '\n' + note


def get_words():
    res = ['Wubba lubba dub dub', 'haha']

    return res[0]


class MailLogs:
    def __init__(self):
        pass

    def sendmail(self):
        pass

    def set_to(self, to_addr):
        self.to_addr = to_addr


if __name__ == '__main__':

    import time

    QQmail = MailLogs()
    QQmail.sendmail()

    while 1:
        pass

