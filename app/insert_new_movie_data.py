#Importing the modules
from os import path as osp
import requests

import json

import os

import pandas as pd
from torch_geometric.datasets import MovieLens

apikey = ''
key1 = 'e760129c'
key2 = 'e44e5305'
key3 = '8403a97b'
key4 = '192c6b0e'

root = osp.join('.', 'tmp', 'ml')
data = MovieLens(root=root, name='1m', num_core=10).data
movies = data.items[0]

director_list = []
actor_list = []

for i, (title, year) in enumerate(zip(movies.title, movies.year)):

   if i in range(0,1000):
      apikey = key1
   if i in range(1000,2000):
      apikey = key2
   if i in range(2000,3000):
      apikey = key3
   if i in range(3000,4000):
      apikey = key4

   movie_url = "http://www.omdbapi.com/?" + "t=" + title + "&y=" + str(year) + "&apikey=" + apikey
   # print('i=' + str(i) + ',apikey=' + apikey )
   try:
      r = requests.get(movie_url)
      movie_info_dic = json.loads(r.text)
      director = movie_info_dic.get('Director', '')
      actor = movie_info_dic.get('Actors', '')
   except:
      director = ''
      actor = ''

   print('i=' + str(i) + ',title = '+ title + ',year = ' + str(year) + ',director = ' + director)
   director_list.append(director)
   actor_list.append(actor)

movies['Director'] = director_list
movies['Actor'] = actor_list
movies.to_csv('new_movies.dat')
# import pdb
# pdb.set_trace()