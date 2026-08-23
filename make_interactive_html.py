import pickle
import json
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

clouds = ['cube', 'sphere', 'wind', 'era5']

tasks = ['compression', 'denoising']

kernels = ['default', 'adjusted']

parameters = {
    'cube': [
        'eps_0.1_eps_pca_0.05',
        'eps_0.04_eps_pca_0.03'
    ],

    'sphere': [
        'eps_0.1_eps_pca_0.05',
        'eps_0.05_eps_pca_0.04'
    ],

    'wind': [
        'eps_1.5e9_eps_pca_1e9',
        'eps_1.2e9_eps_pca_8.6e8'
    ],

    'era5': [
        'eps_0.035_eps_pca_0.03'
    ]
}


# ============================================================
# Location
# ============================================================

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT / "res"

OUTPUT_FILE = ROOT / "interactive_results.html"


# ============================================================
# Convert NumPy objects into JSON-compatible objects
# ============================================================

def make_jsonable(obj):

    # numpy scalar / array
    if hasattr(obj, "tolist"):
        return make_jsonable(obj.tolist())

    # dictionary
    if isinstance(obj, dict):
        return {
            str(key): make_jsonable(value)
            for key, value in obj.items()
        }

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [
            make_jsonable(value)
            for value in obj
        ]

    # ordinary scalar
    return obj


# ============================================================
# Load all experiments
# ============================================================

all_data = {}

print("\nLoading experimental data...\n")

for cloud in clouds:

    all_data[cloud] = {}

    for kernel in kernels:

        all_data[cloud][kernel] = {}

        for parameter in parameters[cloud]:

            all_data[cloud][kernel][parameter] = {}

            for task in tasks:

                file_path = (
                    RES_DIR
                    / cloud
                    / parameter
                    / f"{task}_{kernel}_kernel_nmse.pkl"
                )

                if not file_path.exists():

                    print(
                        f"[WARNING] File not found:\n"
                        f"    {file_path}\n"
                    )

                    continue

                print(f"Loading: {file_path}")

                with open(file_path, "rb") as f:
                    nmse = pickle.load(f)

                all_data[cloud][kernel][parameter][task] = (
                    make_jsonable(nmse)
                )


# ============================================================
# Print summary
# ============================================================

print("\nLoaded datasets:")

for cloud in all_data:

    for kernel in all_data[cloud]:

        for parameter in all_data[cloud][kernel]:

            for task in all_data[cloud][kernel][parameter]:

                print(
                    f"  {cloud} | "
                    f"{kernel} | "
                    f"{parameter} | "
                    f"{task}"
                )


# ============================================================
# Convert everything to JSON
# ============================================================

json_data = json.dumps(
    all_data,
    separators=(",", ":")
)


print(
    f"\nEmbedded JSON size: "
    f"{len(json_data) / 1024 / 1024:.2f} MB"
)


# ============================================================
# HTML
# ============================================================

html = r'''
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Interactive Signal Processing Results
</title>


<!-- Plotly -->

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>


<style>

body {

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    margin: 30px;

    background: #fafafa;

    color: #222;

}


h1 {

    margin-bottom: 5px;

}


.subtitle {

    color: #666;

    margin-bottom: 25px;

}


.tabs {

    margin-bottom: 20px;

}


button {

    padding:
        10px 18px;

    margin-right: 5px;

    border:
        1px solid #aaa;

    background:
        white;

    border-radius:
        5px;

    cursor:
        pointer;

    font-size:
        15px;

}


button.active {

    background:
        #333;

    color:
        white;

}


.controls {

    display:
        flex;

    flex-wrap:
        wrap;

    gap:
        14px;

    align-items:
        flex-start;

    padding:
        18px;

    background:
        white;

    border:
        1px solid #ddd;

    border-radius:
        8px;

    margin-bottom:
        20px;

}


.control {

    display:
        flex;

    flex-direction:
        column;

    gap:
        5px;

}


.control label {

    font-weight:
        bold;

    font-size:
        13px;

}


select {

    min-width:
        190px;

    padding:
        7px;

    font-size:
        14px;

}


.laplacians {

    width:
        100%;

    border-top:
        1px solid #ddd;

    padding-top:
        12px;

    display:
        flex;

    gap:
        18px;

    flex-wrap:
        wrap;

}


.laplacian {

    font-weight:
        normal;

    display:
        flex;

    align-items:
        center;

    gap:
        5px;

}


#plot {

    background:
        white;

    border:
        1px solid #ddd;

    border-radius:
        8px;

}


</style>

</head>


<body>


<h1>
Interactive Signal Processing Results
</h1>


<div class="subtitle">
NMSE results for compression and denoising experiments
</div>


<!-- ========================================================
     Task buttons
     ======================================================== -->

<div class="tabs">

<button
    id="compressionButton"
    class="active"
    onclick="setTask('compression')">

    Compression

</button>


<button
    id="denoisingButton"
    onclick="setTask('denoising')">

    Denoising

</button>

</div>


<!-- ========================================================
     Controls
     ======================================================== -->

<div class="controls">


<div class="control">

<label>
Dataset
</label>

<select
    id="dataset"
    onchange="datasetChanged()">
</select>

</div>


<div class="control">

<label>
Kernel
</label>

<select
    id="kernel"
    onchange="updatePlot()">
</select>

</div>


<div class="control">

<label>
Parameters
</label>

<select
    id="parameter"
    onchange="updatePlot()">
</select>

</div>


<div class="control">

<label>
Number of scales
</label>

<select
    id="scales"
    onchange="updatePlot()">
</select>

</div>


<div
    class="control"
    id="atomsControl">

<label>
Number of atoms
</label>

<select
    id="atoms"
    onchange="updatePlot()">
</select>

</div>


<div
    class="laplacians"
    id="laplacians">

</div>


</div>


<!-- ========================================================
     Plot
     ======================================================== -->

<div id="plot"></div>


<script>


// ============================================================
// Embedded data
// ============================================================

const DATA = __DATA__;


// ============================================================
// Configuration
// ============================================================

const CLOUDS = [
    'cube',
    'sphere',
    'wind',
    'era5'
];


const KERNELS = [
    'default',
    'adjusted'
];


const PARAMETERS = {

    cube: [
        'eps_0.1_eps_pca_0.05',
        'eps_0.04_eps_pca_0.03'
    ],

    sphere: [
        'eps_0.1_eps_pca_0.05',
        'eps_0.05_eps_pca_0.04'
    ],

    wind: [
        'eps_1.5e9_eps_pca_1e9',
        'eps_1.2e9_eps_pca_8.6e8'
    ],

    era5: [
        'eps_0.035_eps_pca_0.03'
    ]

};


const LAPLACIANS = [

    'Connection',
    'Connection Normalized',
    'Trivial',
    'Trivial Normalized',
    'Sheaf'

];


const ATOMS = [
    5,
    10,
    25,
    50,
    100,
    200
];


let currentTask = 'compression';


// ============================================================
// Select helper
// ============================================================

function fillSelect(id, values) {

    const select =
        document.getElementById(id);

    select.innerHTML = '';

    values.forEach(value => {

        const option =
            document.createElement('option');

        option.value =
            String(value);

        option.textContent =
            String(value);

        select.appendChild(option);

    });

}


// ============================================================
// Initialize
// ============================================================

function initialize() {

    fillSelect(
        'dataset',
        CLOUDS
    );

    fillSelect(
        'kernel',
        KERNELS
    );

    fillSelect(
        'atoms',
        ATOMS
    );

    createLaplacianControls();

    document.getElementById(
        'dataset'
    ).value = 'cube';

    document.getElementById(
        'kernel'
    ).value = 'default';

    document.getElementById(
        'atoms'
    ).value = '50';

    datasetChanged();

    setTask('compression');

}


// ============================================================
// Laplacian checkboxes
// ============================================================

function createLaplacianControls() {

    const container =
        document.getElementById(
            'laplacians'
        );

    container.innerHTML =
        '<b>Laplacians:</b>';


    LAPLACIANS.forEach(
        (laplacian, index) => {

            const label =
                document.createElement(
                    'label'
                );

            label.className =
                'laplacian';


            const checkbox =
                document.createElement(
                    'input'
                );

            checkbox.type =
                'checkbox';

            checkbox.id =
                'laplacian_' + index;


            // Sheaf off by default

            checkbox.checked =
                laplacian !== 'Sheaf';


            checkbox.addEventListener(
                'change',
                updatePlot
            );


            label.appendChild(
                checkbox
            );


            label.appendChild(
                document.createTextNode(
                    ' ' + laplacian
                )
            );


            container.appendChild(
                label
            );

        }
    );

}


// ============================================================
// Selected Laplacians
// ============================================================

function getSelectedLaplacians() {

    const selected = [];


    LAPLACIANS.forEach(
        (laplacian, index) => {

            const checkbox =
                document.getElementById(
                    'laplacian_' + index
                );


            if (
                checkbox &&
                checkbox.checked
            ) {

                selected.push(
                    laplacian
                );

            }

        }
    );


    return selected;

}


// ============================================================
// Dataset changed
// ============================================================

function datasetChanged() {

    const dataset =
        document.getElementById(
            'dataset'
        ).value;


    // Update parameters

    fillSelect(
        'parameter',
        PARAMETERS[dataset]
    );


    // --------------------------------------------------------
    // Scale ranges
    //
    // Cube / sphere: 2–7
    // Wind / ERA5:   2–9
    // --------------------------------------------------------

    let maxScale;


    if (
        dataset === 'cube' ||
        dataset === 'sphere'
    ) {

        maxScale = 7;

    }
    else {

        maxScale = 9;

    }


    const scales = [];


    for (
        let s = 2;
        s <= maxScale;
        s++
    ) {

        scales.push(s);

    }


    fillSelect(
        'scales',
        scales
    );


    // Default scale

    document.getElementById(
        'scales'
    ).value = '3';


    updatePlot();

}


// ============================================================
// Task
// ============================================================

function setTask(task) {

    currentTask = task;


    document
        .getElementById(
            'compressionButton'
        )
        .classList.toggle(
            'active',
            task === 'compression'
        );


    document
        .getElementById(
            'denoisingButton'
        )
        .classList.toggle(
            'active',
            task === 'denoising'
        );


    document
        .getElementById(
            'atomsControl'
        )
        .style.display =
            task === 'denoising'
            ? 'flex'
            : 'none';


    updatePlot();

}


// ============================================================
// Statistics
// ============================================================

function mean(values) {

    if (
        !values ||
        values.length === 0
    ) {

        return NaN;

    }


    return values.reduce(
        (sum, value) =>
            sum + Number(value),
        0
    ) / values.length;

}


function standardDeviation(values) {

    if (
        !values ||
        values.length <= 1
    ) {

        return 0;

    }


    const m =
        mean(values);


    const variance =
        values.reduce(
            (sum, value) =>
                sum +
                (
                    Number(value) - m
                ) ** 2,
            0
        )
        /
        (
            values.length - 1
        );


    return Math.sqrt(
        variance
    );

}


function confidenceInterval(values) {

    if (
        !values ||
        values.length <= 1
    ) {

        return 0;

    }


    return (
        1.96 *
        standardDeviation(values) /
        Math.sqrt(values.length)
    );

}


// ============================================================
// Plot
// ============================================================

function updatePlot() {

    const dataset =
        document.getElementById(
            'dataset'
        ).value;


    const kernel =
        document.getElementById(
            'kernel'
        ).value;


    const parameter =
        document.getElementById(
            'parameter'
        ).value;


    const scale =
        document.getElementById(
            'scales'
        ).value;


    const selectedLaplacians =
        getSelectedLaplacians();


    console.log(
        'Plot:',
        currentTask,
        dataset,
        kernel,
        parameter,
        scale,
        selectedLaplacians
    );


    // --------------------------------------------------------
    // Get correct experiment
    // --------------------------------------------------------

    const experiment =
        DATA?.[dataset]
            ?.[kernel]
            ?.[parameter]
            ?.[currentTask];


    if (!experiment) {

        console.error(
            'Experiment not found',
            {
                dataset,
                kernel,
                parameter,
                task: currentTask
            }
        );


        Plotly.purge(
            'plot'
        );


        document.getElementById(
            'plot'
        ).innerHTML =
            '<p style="padding:30px">' +
            'No data found for this combination.' +
            '</p>';


        return;

    }


    const traces = [];


    // --------------------------------------------------------
    // Each Laplacian
    // --------------------------------------------------------

    selectedLaplacians.forEach(
        laplacian => {

            if (
                !experiment[laplacian]
            ) {

                console.warn(
                    'Missing laplacian:',
                    laplacian
                );

                return;

            }


            const scaleData =
                experiment[
                    laplacian
                ][
                    String(scale)
                ];


            if (!scaleData) {

                console.warn(
                    'Missing scale:',
                    scale,
                    laplacian
                );

                return;

            }


            let x = [];
            let y = [];
            let ci = [];


            // =================================================
            // Compression
            // =================================================

            if (
                currentTask ===
                'compression'
            ) {

                const Ks =
                    Object.keys(
                        scaleData
                    )
                    .map(Number)
                    .sort(
                        (a, b) => a - b
                    );


                Ks.forEach(K => {

                    const values =
                        scaleData[
                            String(K)
                        ];


                    if (
                        !Array.isArray(
                            values
                        )
                    ) {

                        return;

                    }


                    x.push(K);

                    y.push(
                        mean(values)
                    );

                    ci.push(
                        confidenceInterval(
                            values
                        )
                    );

                });

            }


            // =================================================
            // Denoising
            // =================================================

            else {

                const numAtoms =
                    document.getElementById(
                        'atoms'
                    ).value;


                /*
                 *
                 * scale
                 *   |
                 *   +-- SNR
                 *        |
                 *        +-- number of atoms
                 *              |
                 *              +-- signal NMSEs
                 *
                 */


                const SNRs =
                    Object.keys(
                        scaleData
                    )
                    .map(Number)
                    .sort(
                        (a, b) => a - b
                    );


                SNRs.forEach(snr => {

                    const snrData =
                        scaleData[
                            String(snr)
                        ];


                    if (!snrData) {

                        return;

                    }


                    const values =
                        snrData[
                            String(numAtoms)
                        ];


                    if (
                        !Array.isArray(
                            values
                        )
                    ) {

                        return;

                    }


                    x.push(snr);

                    y.push(
                        mean(values)
                    );

                    ci.push(
                        confidenceInterval(
                            values
                        )
                    );

                });

            }


            // -------------------------------------------------
            // No data
            // -------------------------------------------------

            if (x.length === 0) {

                console.warn(
                    'No values:',
                    laplacian,
                    scale
                );

                return;

            }


            // -------------------------------------------------
            // Main curve
            // -------------------------------------------------

            traces.push({

                x: x,

                y: y,

                mode:
                    'lines+markers',

                type:
                    'scatter',

                name:
                    laplacian

            });


            // -------------------------------------------------
            // Confidence interval
            // -------------------------------------------------

            const upper =
                y.map(
                    (value, i) =>
                        value + ci[i]
                );


            const lower =
                y.map(
                    (value, i) =>
                        value - ci[i]
                );


            traces.push({

                x:
                    x.concat(
                        [...x].reverse()
                    ),

                y:
                    upper.concat(
                        [...lower].reverse()
                    ),

                fill:
                    'toself',

                fillcolor:
                    'rgba(100,100,100,0.12)',

                line: {
                    color:
                        'rgba(255,255,255,0)'
                },

                hoverinfo:
                    'skip',

                showlegend:
                    false,

                type:
                    'scatter'

            });

        }
    );


    // ========================================================
    // Title
    // ========================================================

    let title;


    if (
        currentTask ===
        'compression'
    ) {

        title =
            'NMSE vs Number of Non-Zero Coefficients' +
            '<br><sup>' +
            dataset +
            ' | ' +
            kernel +
            ' | ' +
            parameter +
            ' | ' +
            scale +
            ' scales' +
            '</sup>';

    }
    else {

        const numAtoms =
            document.getElementById(
                'atoms'
            ).value;


        title =
            'NMSE vs Input SNR' +
            '<br><sup>' +
            dataset +
            ' | ' +
            kernel +
            ' | ' +
            parameter +
            ' | ' +
            scale +
            ' scales' +
            ' | ' +
            numAtoms +
            ' atoms' +
            '</sup>';

    }


    // ========================================================
    // Layout
    // ========================================================

    const layout = {

        title:
            title,

        xaxis: {

            title:
                currentTask ===
                'compression'
                ? 'Number of non-zero coefficients'
                : 'Input SNR',

            type:
                currentTask ===
                'compression'
                ? 'linear'
                : 'log',

            gridcolor:
                '#dddddd'

        },


        yaxis: {

            title:
                'NMSE',

            type:
                'log',

            gridcolor:
                '#dddddd'

        },


        hovermode:
            'x unified',

        template:
            'plotly_white',

        height:
            650,

        margin: {

            l: 80,

            r: 40,

            t: 100,

            b: 80

        }

    };


    // ========================================================
    // IMPORTANT:
    // Plotly.react updates an existing plot rather than
    // creating a new one.
    // ========================================================

    Plotly.react(
        'plot',
        traces,
        layout,
        {
            responsive: true,
            displaylogo: false
        }
    );

}


// ============================================================
// Start application
// ============================================================

initialize();


</script>

</body>

</html>
'''


# ============================================================
# Insert JSON into HTML
# ============================================================

html = html.replace(
    "__DATA__",
    json_data
)


# ============================================================
# Write HTML
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(html)


print("\n========================================")
print("Interactive HTML successfully created!")
print("========================================")
print()
print(f"File: {OUTPUT_FILE}")
print()
print("Open it directly in Chrome.")