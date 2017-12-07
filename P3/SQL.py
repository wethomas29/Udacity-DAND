#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 21:03:48 2017

@author: williamthomas
"""

import sqlite3
db = sqlite3.connect("Nola.db")
c = db.cursor()

c.execute('''
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY NOT NULL,
    lat REAL,
    lon REAL,
    user TEXT,
    uid INTEGER,
    version INTEGER,
    changeset INTEGER,
    timestamp TEXT
);
''')

c.execute('''
CREATE TABLE nodes_tags (
    id INTEGER,
    key TEXT,
    value TEXT,
    type TEXT,
    FOREIGN KEY (id) REFERENCES nodes(id)
);
''')

c.execute('''
CREATE TABLE ways (
    id INTEGER PRIMARY KEY NOT NULL,
    user TEXT,
    uid INTEGER,
    version TEXT,
    changeset INTEGER,
    timestamp TEXT
);
''')

c.execute('''
CREATE TABLE ways_tags (
    id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    type TEXT,
    FOREIGN KEY (id) REFERENCES ways(id)
);
''')

c.execute('''
CREATE TABLE ways_nodes (
    id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    FOREIGN KEY (id) REFERENCES ways(id),
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);
''')

db.commit()

# Read in the csv file as a dictionary, format the data as a list of tuples:
with open('nodes.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['lat'], i['lon'], i['user'].decode("utf-8"), i['uid'], i['version'], i['changeset'], i['timestamp']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO nodes(id, lat, lon, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", to_db)
# commit the changes
db.commit()

with open('nodes_tags.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['key'], i['value'].decode("utf-8"), i['type']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO nodes_tags(id, key, value,type) VALUES (?, ?, ?, ?);", to_db)
# commit the changes
db.commit()

with open('ways.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['user'].decode("utf-8"), i['uid'], i['version'], i['changeset'], i['timestamp']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO ways(id, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?);", to_db)
# commit the changes
db.commit()


with open('ways_nodes.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['node_id'], i['position']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO ways_nodes(id, node_id, position) VALUES (?, ?, ?);", to_db)
# commit the changes
db.commit()

with open('ways_tags.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['key'], i['value'].decode("utf-8"), i['type']) for i in dr]
    
# insert the formatted data
c.executemany("INSERT INTO ways_tags(id, key, value, type) VALUES (?, ?, ?, ?);", to_db)
# commit the changes
db.commit()


import sqlite3
db = sqlite3.connect("Nola.db")
c = db.cursor()

#Number of Ways
query = "SELECT count(*) FROM ways;"
c.execute(query)
c.fetchall()[0][0]

#Number of Common Way Tags (Top 5)
query = "SELECT key, count(*) FROM ways_tags GROUP BY 1 ORDER BY count(*) DESC LIMIT 5;"
c.execute(query)
c.fetchall()

#Number of Nodes
query = "SELECT count(*) FROM nodes;"
c.execute(query)
c.fetchall()[0][0]

#Number of Common Node Tags
query = "SELECT key,count(*) FROM nodes_tags GROUP BY 1 ORDER BY count(*) DESC LIMIT 5;"
c.execute(query)
c.fetchall()

#Contributors
query = "SELECT temp.user, count(*) as posts FROM (SELECT user, uid FROM ways UNION ALL SELECT user, uid FROM nodes) as temp \
GROUP BY temp.user ORDER BY posts DESC LIMIT 10;"
c.execute(query)
c.fetchall()

#Top Users
query = "SELECT count(DISTINCT(temp.uid)) FROM (SELECT user, uid FROM ways UNION ALL SELECT user, uid FROM nodes) as temp;"
c.execute(query)
c.fetchall()[0][0]

#Top 10 Amenities
query = "SELECT temp.value, count(*) as num \
FROM (SELECT key,value FROM ways_tags UNION ALL SELECT key,value FROM nodes_tags) as temp \
WHERE temp.key='amenity' GROUP BY temp.value ORDER BY num DESC LIMIT 10;"
c.execute(query)
c.fetchall()

#Top 10 Cuisines
query = "SELECT temp.value, count(*) as num \
FROM (SELECT key,value FROM ways_tags UNION ALL SELECT key,value FROM nodes_tags) as temp \
WHERE temp.key='cuisine' GROUP BY temp.value ORDER BY num DESC LIMIT 10;"
c.execute(query)
c.fetchall()

