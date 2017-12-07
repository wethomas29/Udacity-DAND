#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:42:39 2017

@author: williamthomas
"""

# Creating sample file as original OSM is 1.28 GB unzipped.
# Parameter: take every k-th top level element
#Sample size is 128.9 MB

OSM_FILE = "new-orleans_louisiana.osm"
SAMPLE_FILE = "new-orleans_louisiana_sample.osm"

k = 10

def get_element(osm_file, tags=('node', 'way', 'relation')):
    '''Yield element if it is the right type of tag

    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    '''
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


with open(SAMPLE_FILE, 'wb') as output:
    output.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write(b'<osm>\n  ')

    # Write every kth top level element
    for i, element in enumerate(get_element(OSM_FILE)):
        if i % k == 0:
            output.write(ET.tostring(element, encoding='utf-8'))

    output.write(b'</osm>')