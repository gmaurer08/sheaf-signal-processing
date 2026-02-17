import numpy as np

# Function that computes a Fibonacci Sphere
def fibonacci_sphere(N):
    i = np.arange(N)
    phi = (1+np.sqrt(5))/2
    z = 1-2*i/(N-1)
    theta = 2*np.pi*i/phi
    r = np.sqrt(1-z*z)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.column_stack((x,y,z))

# Function that turns lat_deg, lon_deg location data into x,y,z coordinates in R^3 (on the unit sphere)
def geodetic_to_ecef(lat_deg, lon_deg, height_m):
    """
    Convert latitude, longitude, height to ECEF coordinates (WGS84).

    Parameters
    ----------
    lat_deg : float
        Latitude in degrees
    lon_deg : float
        Longitude in degrees
    height_m : float
        Height above ellipsoid in meters

    Returns
    -------
    x, y, z : tuple of floats
    """
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)

    N = a / np.sqrt(1 - e2 * sin_lat**2)

    #x = (N + height_m) * cos_lat * np.cos(lon)
    #y = (N + height_m) * cos_lat * np.sin(lon)
    #z = (N * (1 - e2) + height_m) * sin_lat

    x = cos_lat * np.cos(lon)
    y = cos_lat * np.sin(lon)
    z = sin_lat

    return x, y, z


def project_to_tangent(V, E1, E2):
    """
    Project 3D vectors onto local tangent planes.

    Parameters
    ----------
    V : (N, 3) array
        Vectors in R^3 (ECEF)
    E1 : (N, 3) array
        First tangent basis vector (unit)
    E2 : (N, 3) array
        Second tangent basis vector (unit)

    Returns
    -------
    V_tan : (N, 2) array
        Coordinates in local tangent basis
    """
    v1 = np.einsum("ij,ij->i", V, E1)
    v2 = np.einsum("ij,ij->i", V, E2)

    return np.column_stack((v1, v2))
