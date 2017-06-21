#!/usr/bin/python

import csv, sys, argparse, re, collections
import os
import pathlib

category = []
title = []
content = []
streamCatego = []
my_dict = {}
stream_dict = {}

# definition functions
def makefolder(strings):
	if not os.path.exists(strings):
		#os.makedirs(strings)
		pathlib.Path(strings).mkdir(parents=True, exist_ok=True)
	return

def writeText(strings, category, key, content):
	path = strings + category + '/' + key
	with open(path, 'wb') as x_file:
		x_file.write('{}'.format(content))
	

# read file csv
#with open(sys.argv[1]) as csvfile:
with open('1e6aa914555236a0c9cea7a330d5a04e_170607144459.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		category.append(row['intent'])
		#title.append(row['title'])
		content.append(row['text'])

# counter class
counter = collections.Counter(category)
strings = list(counter.keys())
values = list(counter.values())

d_string = strings[1]
# create folder
for i in range(len(strings)):
	newpathTrain = '~/train/' + strings[i]
	makefolder(newpathTrain)
	newpathTest = '~/test/' + strings[i]
	makefolder(newpathTest)
	stream_dict[strings[i]] = values[i]

## divide file
for i in range(len(category)):
	key = category[i]
	if key in my_dict:
		my_dict[key] = my_dict.get(key, 0) + 1
		if (my_dict.get(key, 0) <= (stream_dict.get(key, 0) * 0.8)):
			writeText('~/train/',category[i], str(my_dict.get(key, 0)), content[i])
		else:
			writeText('~/test/',category[i], str(my_dict.get(key, 0)), content[i])
	else:
		my_dict[key] = 0