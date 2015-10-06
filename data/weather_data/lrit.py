__author__ = 'masayuki'

import math

'''
Utilties for LRIT / HRIT Satellite images
'''

def lonlat2xy(prj_dir, prj_lon, lon, lat):
    '''
    project geographical coordinates (lon,lat) to intermediate coordinates (x,y)
    :param prj_dir: the projection plane, 'N' or 'S'
    :param prj_lon: the central longitude in degrees
    :param lat: latitude in degrees
    :param lon: longitude in degrees
    :return: a tuple of (x, y)
    '''
    assert prj_dir in ('N', 'S')

    lon = math.radians(lon)
    lat = math.radians(lat)

    d = 1 if prj_dir == 'N' else -1
    x = math.tan((math.pi/2.0 - d*lat)/2.0) * math.sin(lon - prj_lon)
    y = math.tan((math.pi/2.0 - d*lat)/2.0) * math.cos(lon - prj_lon)

    return (x, y)

def xy2lonlat(prj_dir, prj_lon, x, y):
    '''
    project intermediate coordinates (x,y) to geographical coordinates (lon,lat)
    :param prj_dir: the projection plane, 'N' or 'S'
    :param prj_lon: the central longitude
    :param x: x
    :param y: y
    :return: a tuple of (lon, lat) in degrees
    '''
    assert prj_dir in ('N', 'S')

    d = 1 if prj_dir == 'N' else -1
    sy = 1 if y > 0.0 else 0 if y == 0.0 else -1
    lon = math.atan(x/y) + prj_lon + d*(math.pi/2.0)*(1-sy)
    lat = d*(math.pi/2.0 - 2*math.atan(math.sqrt(x**2+y**2)))

    lon = math.degrees(lon)
    lat = math.degrees(lat)

    return (lon, lat)

def xy2cl(cfac, lfac, coff, loff, x, y):
    '''
    convert intermediate coordinates (x,y) to image coordinate (c,l)
    :param cfac: CFAC
    :param lfac: LFAC
    :param coff: COFF
    :param loff: LOFF
    :param x: x
    :param y: y
    :return:
    '''
    nint = lambda r: int(round(r))
    c =  nint(x * 2**(-16) * cfac) + coff
    l =  nint(y * 2**(-16) * lfac) + loff

    return (c, l)

def cl2xy(cfac, lfac, coff, loff, c, l):
    '''
    convert intermediate coordinates (x,y) to image coordinate (c,l)
    :param cfac: CFAC
    :param lfac: LFAC
    :param coff: COFF
    :param loff: LOFF
    :param c: c
    :param l: l
    :return:
    '''
    x = 2**16 * (c - coff) / cfac
    y = 2**16 * (l - loff) / lfac

    return (x, y)