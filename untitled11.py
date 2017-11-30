# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:28:47 2017

@author: Abhishek_Mishra9
"""

import shutil
import requests
import json

proxy = {
'user' : 'user',
'pass' : 'password',
'host' : "test.net",
'port' : 8080
}

url = 'http://tools.dell.com/GCM/gcmp.management/Audits/GetOrderGroup?DPId=2005484072209'
response = requests.get(url,verify=True)

with open(r'..\test.json','wb') as out_file:
      out_file.write(response.text)
print (response)