#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:39:19 2017

@author: williamthomas
"""
import xml.etree.cElementTree as ET
from pprint import pprint
import re
from collections import defaultdict
import csv
import codecs
import cerberus


# Load the sample OSM file as provided in GitHub repository
OSMFILE = "new-orleans_louisiana_sample.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road"]

# Create a group of auditing functions for street suffix
def audit_street_type(street_types, street_name):
    """ Checks if street name contains incorrect abbreviations, if so, adds it to the dictionary. """
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    """ Returns a Boolean value """
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    """ Iterates through document tags, and returns dictionary
        of incorrect abbreviations (keys) and street names (value) that contain these abbreviations.
    """
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types

# Run audit and print results
st_types = audit(OSMFILE)
pprint(dict(st_types))


# Function to correct street names using wrong suffix
def update_name(name, mapping):
    """ Substitutes incorrect abbreviation with correct one. """
    m = street_type_re.search(name)
    if m:
        street_type = m.group()
        
        temp= 0
        try:
            temp = int(street_type)
        except:
            pass
        
        if street_type not in expected and temp == 0:
            try:
                name = re.sub(street_type_re, mapping[street_type], name)
            except:
                pass
    return name


# Dictionary mapping incorrect abbreviations to correct one.
mapping = { "St": "Street",
            "St.": "Street",
            "ST": "Street",
            "st": "Street",
            "Rd.": "Road",
            "Rd": "Road",
            "RD": "Road",
            "Ave": "Avenue",
            "Ave.": "Avenue",
            "Blvd": "Boulevard",
            "BLVD": "Boulevard",
            "Cir": "Circle",
            "Ct": "Court",
            "Dr": "Drive",
            "Trl": "Trail",
            "Ter": "Terrace",
            "Pl": "Place",
            "Pkwy": "Parkway",
            "Bnd": "Bend",
            "Mnr": "Manor",
            "Ln": "Lane",
            "street": "Street",
            "AVE": "Avenue",
            "Blvd.": "Boulevard",
            "Cirlce": "Circle",
            "DRIVE": "Drive",
            "Cv": "Cove",
            "Dr.": "Drive",
            "Druve": "Drive",
            "Holw": "Hollow",
            "Hwy": "Highway",
            "HWY": "Highway",
            "Pt": "Point",
            "Trce": "Trace",
            "ave": "Avenue",
            "Cres": "Crescent"
            }

# Apply corrections where incorrect detected v. mapping.
for st_type, ways in st_types.iteritems():
    for name in ways:
        better_name = update_name(name, mapping)
        print name, "=>", better_name

################################
'POSTAL CODE AUDIT'
 ############################      
# # Create a group of auditing functions for postal codes

def audit_postcode(post_code, digits):
    """ Checks if postal code is incompatible and adds it to the list if so. """
    if len(digits) != 5 or (digits[0:2] != '01' and digits[0:2] != '02'):
        post_code.append(digits)
        
def is_postalcode(elem):
    """ Returns a Boolean value."""
    return (elem.attrib['k'] == "addr:postcode")


def audit(osmfile):
    """ Iterates and returns list of inconsistent postal codes found in the document. """
    osm_file = open(osmfile, "r")
    post_code = []
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_postalcode(tag):
                    audit_postcode(post_code, tag.attrib['v'])
    osm_file.close()
    return post_code

# Run audit and print results
postal_codes = audit(OSMFILE)
print postal_codes

# Apply corrections where incorrect detected (No problems with Zip Code)
#for code in postal_codes:
#    better_code = update_zip(code)
#    print code, "=>", better_code
