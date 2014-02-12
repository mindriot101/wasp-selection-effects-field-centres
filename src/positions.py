#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from glob import glob
import csv
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from astropy.io import fits as pyfits
import itertools
import re
from astropy.coordinates import Galactic, ICRS
import astropy.units as u
import logging
from planetdatabase.waspdatabase import PublishedWASPDatabase as WASPDatabase
import threading
from multiprocessing.dummy import Pool as ThreadPool


logging.basicConfig(level=logging.INFO)
REGEX = re.compile(r'MSSW(?P<ra>\d{4})(?P<dec>[-+]\d{4})')

base_path = os.path.join(
        os.path.dirname(__file__),
        '..')

data_dir = os.path.join(base_path, 'data')
output_dir = os.path.join(base_path, 'output')

NORTH_COLOUR = '#009999'
SOUTH_COLOUR = '#9FEE00'
PLANET_COLOUR = '#FF0000'

def get_files():
    '''
    Given the MS*.fits files in the data directory, return this list
    given that each filename must have a unique field.
    '''
    known_coords = set([])
    for fname in glob(os.path.join(data_dir, 'MS*.fits')):
        basename = os.path.basename(fname)
        match = REGEX.match(basename)

        coords = ''.join([match.group(i) for i in [1, 2]])

        if coords not in known_coords:
            known_coords.add(coords)
            yield fname
        else:
            logging.debug('Duplicate found: {}'.format(basename))

def ra_to_radians(ra):
    hour, minute = int(ra[:2]), int(ra[2:])
    degrees = (hour + minute / 60.0) * 15.
    return np.radians(degrees)

def dec_to_radians(dec):
    sign, degrees, minutes = dec[0], int(dec[1:3]), int(dec[3:])
    total = degrees + minutes / 60.
    full = -total if sign == '-' else total
    return np.radians(full)
    

def is_south_field(camera_id):
    is_south = 200 <= int(camera_id) < 300
    if logging.getLogger().getEffectiveLevel() >= logging.DEBUG:
        if is_south:
            logging.debug('Camera {} is south'.format(camera_id))
        else:
            logging.debug('Camera {} is north'.format(camera_id))
    return is_south

def construct_rectangle(ra, dec, field_size_degrees, *args, **kwargs):
    half_size = field_size_degrees / 2.
    half_size_radians = np.radians(half_size)
    field_size_radians = np.radians(field_size_degrees)

    return plt.Rectangle((ra - half_size_radians, dec - half_size_radians),
            field_size_radians, field_size_radians, *args, **kwargs)



def field_rectangle((ra, dec, camera_id), field_size_degrees, *args, **kwargs):
    '''
    Constructs the matplotlib `Rectangle` of a single WASP field
    '''
    if is_south_field(camera_id):
        colour = SOUTH_COLOUR
    else:
        colour = NORTH_COLOUR

    return construct_rectangle(ra, dec, field_size_degrees, color=colour)

def calculate_field_size():
    '''
    Returns the WASP field size from the single number I was given
    '''
    square_degrees = 64.
    logging.info("Using field size of {} square degrees".format(square_degrees))
    return np.sqrt(square_degrees)

def galactic_plane():
    '''
    Computes the equatorial coordinates of the galactic plane
    '''
    logging.info("Constructing galactic plane")
    l_radians = np.radians(np.linspace(0, 360, 500))
    b_radians = np.zeros_like(l_radians)

    eq_coords = Galactic(l_radians, b_radians, unit=(u.radian, u.radian)).icrs
    ra, dec = np.pi - eq_coords.ra.value, eq_coords.dec.value

    # sort in order of ra
    ind = np.argsort(ra)
    return ra[ind], dec[ind]

def plot_galactic_plane(ax):
    '''
    Plots the galactic plane onto axis `ax`
    '''
    gal_ra, gal_dec = galactic_plane()
    ax.plot(gal_ra, gal_dec, 'k-')

wasp_regex = re.compile(r'WASP-?(\d+)')
def format_wasp_name(name):
    try:
        return int(wasp_regex.search(name).group(1))
    except AttributeError:
        pass


def known_planets():
    '''
    Returns the known wasp planets which were included in our dataset
    '''
    wd = WASPDatabase()

    names = set([format_wasp_name(row[0]) for row in wd.data()])
    included = set([])

    with open(os.path.join(data_dir, 'wasp_planets.csv')) as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            name = row['# name']

            include_by_name = (name.startswith('WASP') or 
                    'WASP-40' in name 
                    or 'WASP-51' in name)

            if include_by_name and format_wasp_name(name) in names:
                included.add(format_wasp_name(name))
                ra = float(row[' ra'])
                dec = float(row[' dec'])

                if ra != 0.0 and dec != 0.0:
                    yield np.radians(180 - ra), np.radians(dec)

    logging.debug('Not included planets: {}'.format(names - included))


def plot_known_planets(ax):
    coords = known_planets()
    ra, dec = zip(*coords)
    ax.plot(ra, dec, color=PLANET_COLOUR, ls='None', marker='o', zorder=10)

def extract_field_info(fname):
    logging.info("Extracting information for [{}]".format(fname))
    match = REGEX.search(fname)
    ra = np.pi - ra_to_radians(match.group('ra'))
    dec = dec_to_radians(match.group('dec'))
    with pyfits.open(fname) as infile:
        camera_id = infile[0].header['camera_id']
    return ra, dec, camera_id

def extract_all_info(files):
    cache_filename = os.path.join(data_dir, 'cache.txt')

    if os.path.isfile(cache_filename):
        logging.info("Reading from cache")
        with open(cache_filename) as infile:
            for line in infile:
                ra, dec, camera_id = line.strip('\n').split()
                yield float(ra), float(dec), int(camera_id)
    else:
        logging.info("Computing data")
        pool = ThreadPool()
        with open(cache_filename, 'w') as outfile:
            for ra, dec, camera_id in pool.imap(extract_field_info, files):
                outfile.write('{} {} {}\n'.format(ra, dec, camera_id))
                yield ra, dec, camera_id

def plot_latitudes(ax):
    '''
    Plot markers for the latitudes of the two observatories
    '''
    north_lat = np.radians(28.667)
    south_lat = np.radians(-33.9347)

    for lat in north_lat, south_lat:
        plt.plot([-np.pi, np.pi], [lat, lat], 'k--')

def plot_kepler_field(ax):
    centre = ICRS('19h22m40s +44d30m00s')
    logging.info('Plotting kepler field at {}'.format(centre))
    field_size = np.sqrt(105 * u.degree * u.degree).value

    ra, dec = centre.ra.radian, centre.dec.radian
    field = construct_rectangle(np.pi - ra, dec, field_size, color='k')
    ax.add_patch(field)


    
def main():
    field_size = calculate_field_size()
    files = get_files()
    field_information = extract_all_info(files)

    rectangles = (field_rectangle(field_info, field_size, alpha=0.7)
            for field_info in field_information)

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection='mollweide')
    for r in rectangles:
        ax.add_patch(r)

    plot_latitudes(ax)
    plot_galactic_plane(ax)
    plot_kepler_field(ax)
    plot_known_planets(ax)

    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fields.png'), bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
