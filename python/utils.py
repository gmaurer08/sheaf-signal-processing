import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from python.tsp import TSP
from tqdm import tqdm

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

    x = (N + height_m) * cos_lat * np.cos(lon)
    y = (N + height_m) * cos_lat * np.sin(lon)
    z = (N * (1 - e2) + height_m) * sin_lat

    #x = cos_lat * np.cos(lon)
    #y = cos_lat * np.sin(lon)
    #z = sin_lat

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



# Function that builds a local refernce frame given lat, lon coordinates in degrees
def local_reference_frame(lat_deg, lon_deg):

    # Convert degrees to radians
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    # Create local 3D reference frame
    e_E = np.array([-np.sin(lon), np.cos(lon), 0.0]) # East
    e_N = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]) # North
    e_U = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]) # Up (radial)

    return e_E, e_N, e_U

# Function that, given wind direction and location, computes the 3D coordinates of the wind in the local reference frame
def wind_uv_to_xyz(u, v, lat_deg, lon_deg):

    # Get the local reference frame directions
    e_E, e_N, e_U = local_reference_frame(lat_deg, lon_deg)

    # Compute the 3D wind
    wind_xyz = u * e_E + v * e_N

    return wind_xyz


# Function that turns a defaultdict into a dictionary
def dictify(d):
    if isinstance(d, defaultdict):
        d = {k: dictify(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: dictify(v) for k, v in d.items()}
    return d


# function to find the best eps and eps_pca
def find_best_eps_eps_pca(points, eps_list, eps_pca_list, k, gamma):
    working_tuples = []
    for i, eps in tqdm(enumerate(eps_list)):
        for j, eps_pca in tqdm(enumerate(eps_pca_list[:i])):
            try:
                obj = TSP(points, eps=eps, eps_pca=eps_pca, k=k, gamma=gamma)
                dim = obj.estimate_dim()
                if dim==2:
                    working_tuples.append((eps, eps_pca))
            except Exception as e:
                #print('failed for eps:', eps, 'and eps_pca:', eps_pca)
                #raise
                pass
    # find best tuple with min eps, min eps_pca
    #print(working_tuples)
    best_tuple = min(working_tuples, key=lambda x: (x[0], x[1]))
    return best_tuple




###############################

###### Interactive Plot


import pickle
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
from scipy.stats import t


clouds = ['cube', 'sphere', 'wind', 'era5']
kernels = ['default', 'adjusted']
parameters = {
    'cube': ['eps_0.1_eps_pca_0.05','eps_0.04_eps_pca_0.03'],
    'sphere': ['eps_0.1_eps_pca_0.05','eps_0.05_eps_pca_0.04'],
    'wind': ['eps_1.5e9_eps_pca_1e9','eps_1.2e9_eps_pca_8.6e8'],
    'era5': ['eps_0.035_eps_pca_0.03']
}
laplacians = ['Connection','Connection Normalized','Trivial', 'Trivial Normalized','Sheaf']

def load_nmse_results(cloud, task, kernel, parameter):
    """
    Load one NMSE pickle file.
    """
    file_path = (
        f'res/{cloud}/{parameter}/'
        f'{task}_{kernel}_kernel_nmse.pkl'
    )

    with open(file_path, 'rb') as f:
        return pickle.load(f)


def interactive_compression_plot():

    # WIDGETS
    cloud_widget = widgets.Dropdown(options=clouds,value=clouds[0],description='Dataset:')
    parameter_widget = widgets.Dropdown(options=parameters[clouds[0]],value=parameters[clouds[0]][0],description='Parameter:')
    kernel_widget = widgets.Dropdown(options=kernels,value=kernels[0],description='Kernel:')

    # Cube is the initial dataset, so start with 2–7
    scale_widget = widgets.Dropdown(options=list(range(2, 8)), value=3, description='Scales:')
    laplacian_widgets = {lap: widgets.Checkbox(value=(lap != 'Sheaf'),description=lap,indent=False) for lap in laplacians}

    # Update parameters and scales when dataset changes
    def update_parameters(change):

        new_cloud = change['new']

        # Update parameter choices
        parameter_widget.options = parameters[new_cloud]
        parameter_widget.value = parameters[new_cloud][0]

        # Update scale choices
        if new_cloud in ['cube', 'sphere']:
            scale_widget.options = list(range(2, 8))   # 2–7
        else:
            scale_widget.options = list(range(2, 10))  # 2–9

        # Make sure current scale is valid
        if scale_widget.value not in scale_widget.options:
            scale_widget.value = scale_widget.options[0]

    cloud_widget.observe(update_parameters, names='value')

    # PLOT
    output = widgets.Output()

    def update_plot(*args):

        with output:

            output.clear_output(wait=True)

            cloud = cloud_widget.value
            parameter = parameter_widget.value
            kernel = kernel_widget.value
            scale = scale_widget.value

            selected_laplacians = [
                lap for lap, widget in laplacian_widgets.items()
                if widget.value
            ]

            # Load data
            nmse_results = load_nmse_results(cloud=cloud,task='compression', kernel=kernel,parameter=parameter)

            fig = go.Figure()

            for laplacian in selected_laplacians:

                if laplacian not in nmse_results:
                    continue

                atom_dict = nmse_results[laplacian][scale]

                Ks = sorted(atom_dict.keys())

                means = []
                ci = []

                for K in Ks:

                    vals = np.asarray(atom_dict[K])

                    mean = np.mean(vals)
                    means.append(mean)

                    n = len(vals)

                    if n > 1:

                        std = np.std(vals, ddof=1)

                        t_value = t.ppf(0.975,df=n - 1)
                        confidence_interval = (t_value * std / np.sqrt(n))

                    else:
                        confidence_interval = 0

                    ci.append(confidence_interval)

                means = np.asarray(means)
                ci = np.asarray(ci)

                # Main curve
                fig.add_trace(
                    go.Scatter(
                        x=Ks,
                        y=means,
                        mode='lines+markers',
                        name=laplacian
                    )
                )

                # Confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=Ks + Ks[::-1],
                        y=list(means + ci)
                        + list((means - ci)[::-1]),
                        fill='toself',
                        fillcolor='rgba(100,100,100,0.12)',
                        line=dict(
                            color='rgba(255,255,255,0)'
                        ),
                        hoverinfo='skip',
                        showlegend=False
                    )
                )

            fig.update_layout(
                title=(
                    'NMSE vs Number of Non-Zero Coefficients'
                    '<br>'
                    f'<sup>{cloud} | {kernel} | '
                    f'{parameter} | {scale} scales</sup>'
                ),

                xaxis_title='Number of non-zero coefficients',
                yaxis_title='NMSE',

                yaxis_type='log',

                template='plotly_white',

                hovermode='x unified',

                width=850,
                height=600
            )

            fig.show()

    # Update plot whenever a widget changes
    cloud_widget.observe(update_plot, names='value')
    parameter_widget.observe(update_plot, names='value')
    kernel_widget.observe(update_plot, names='value')
    scale_widget.observe(update_plot, names='value')

    for widget in laplacian_widgets.values():
        widget.observe(update_plot, names='value')

    # Layout
    controls = widgets.VBox([

        widgets.HBox([cloud_widget, kernel_widget]),
        widgets.HBox([parameter_widget, scale_widget]),

        widgets.HTML('<b>Laplacians:</b>'),

        widgets.HBox([
            widgets.VBox([laplacian_widgets['Connection'],laplacian_widgets['Connection Normalized'],laplacian_widgets['Trivial']]),
            widgets.VBox([laplacian_widgets['Trivial Normalized'],laplacian_widgets['Sheaf']])
        ])
    ])

    display(controls)
    display(output)

    update_plot()

# INTERACTIVE DENOISING PLOT
def interactive_denoising_plot():

    # Widgets
    cloud_widget = widgets.Dropdown(options=clouds, value=clouds[0],description='Dataset:')
    parameter_widget = widgets.Dropdown(options=parameters[clouds[0]],value=parameters[clouds[0]][0],description='Parameter:')
    kernel_widget = widgets.Dropdown(options=kernels, value=kernels[0], description='Kernel:')

    # Cube is the initial dataset, so start with 2–7
    scale_widget = widgets.Dropdown(options=list(range(2, 8)), value=3, description='Scales:')
    atom_widget = widgets.Dropdown(options=[5, 10, 25, 50, 100, 200], value=50, description='Atoms:')

    laplacian_widgets = {lap: widgets.Checkbox(value=(lap != 'Sheaf'),description=lap,indent=False) for lap in laplacians}

    # Update parameters and scales when dataset changes

    def update_parameters(change):

        new_cloud = change['new']

        # Update parameter choices
        parameter_widget.options = parameters[new_cloud]
        parameter_widget.value = parameters[new_cloud][0]

        # Update scale choices
        if new_cloud in ['cube', 'sphere']:
            scale_widget.options = list(range(2, 8))   # 2–7
        else:
            scale_widget.options = list(range(2, 10))  # 2–9

        # Make sure current scale is valid
        if scale_widget.value not in scale_widget.options:
            scale_widget.value = scale_widget.options[0]

    cloud_widget.observe(update_parameters, names='value')

    # Plot
    output = widgets.Output()

    def update_plot(*args):

        with output:

            output.clear_output(wait=True)

            cloud = cloud_widget.value
            parameter = parameter_widget.value
            kernel = kernel_widget.value
            scale = scale_widget.value
            num_atoms = atom_widget.value

            selected_laplacians = [
                lap for lap, widget in laplacian_widgets.items()
                if widget.value
            ]

            # Load data
            nmse_results = load_nmse_results(
                cloud=cloud,
                task='denoising',
                kernel=kernel,
                parameter=parameter
            )

            fig = go.Figure()

            for laplacian in selected_laplacians:

                if laplacian not in nmse_results:
                    continue

                snr_dict = nmse_results[laplacian][scale]
                SNRs = sorted(snr_dict.keys())
                means = []
                ci = []

                for snr in SNRs:

                    vals = np.asarray(snr_dict[snr][num_atoms])
                    mean = np.mean(vals)
                    means.append(mean)
                    n = len(vals)

                    if n > 1:
                        std = np.std(vals, ddof=1)
                        t_value = t.ppf(0.975,df=n - 1)
                        confidence_interval = (t_value * std / np.sqrt(n))
                    else:
                        confidence_interval = 0
                    ci.append(confidence_interval)

                means = np.asarray(means)
                ci = np.asarray(ci)

                # Main curve
                fig.add_trace(
                    go.Scatter(
                        x=SNRs,
                        y=means,
                        mode='lines+markers',
                        name=laplacian
                    )
                )

                # Confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=SNRs + SNRs[::-1],
                        y=list(means + ci)
                        + list((means - ci)[::-1]),
                        fill='toself',
                        fillcolor='rgba(100,100,100,0.12)',
                        line=dict(
                            color='rgba(255,255,255,0)'
                        ),
                        hoverinfo='skip',
                        showlegend=False
                    )
                )

            fig.update_layout(
                title=(
                    'NMSE vs Input SNR'
                    '<br>'
                    f'<sup>{cloud} | {kernel} | '
                    f'{parameter} | '
                    f'{scale} scales | '
                    f'{num_atoms} atoms</sup>'
                ),

                xaxis_title='Input SNR',
                yaxis_title='NMSE',

                xaxis_type='log',
                yaxis_type='log',

                template='plotly_white',

                hovermode='x unified',

                width=850,
                height=600
            )

            fig.show()

    # Update plot whenever a widget changes
    cloud_widget.observe(update_plot, names='value')
    parameter_widget.observe(update_plot, names='value')
    kernel_widget.observe(update_plot, names='value')
    scale_widget.observe(update_plot, names='value')
    atom_widget.observe(update_plot, names='value')

    for widget in laplacian_widgets.values():
        widget.observe(update_plot, names='value')

    # Layout
    controls = widgets.VBox([

        widgets.HBox([cloud_widget, kernel_widget]),
        widgets.HBox([parameter_widget, scale_widget, atom_widget]),
        widgets.HTML('<b>Laplacians:</b>'),
        widgets.HBox([
            widgets.VBox([laplacian_widgets['Connection'], laplacian_widgets['Connection Normalized'],  laplacian_widgets['Trivial']]),
            widgets.VBox([laplacian_widgets['Trivial Normalized'], laplacian_widgets['Sheaf']])])
    ])

    display(controls)
    display(output)

    update_plot()