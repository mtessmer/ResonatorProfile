import argparse
from pathlib import Path
from glob import glob
import numpy as np
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from bokeh.models import HoverTool, Select
from bokeh.layouts import column
from bokeh.plotting import figure, output_file, show, ColumnDataSource
import deerlab as dl

parser = argparse.ArgumentParser(description="    ResonatorProfile automatically reads Bruker BES3T files from the "
                                             "xepr Data directory and "
                                             )
parser.add_argument('destination', type=str, nargs=1,
                    help='Destination directory for experiment files and transfer function')
parser.add_argument('-p', '--prefix', dest='prefix',
                    help='Prefix of nutation files', default='Nut_')
args = parser.parse_args()

destination = Path(args.destination[0])

if not destination.exists():
    raise FileNotFoundError(f'{str(destination)} does not exist. Please reference a valid directory')
if not destination.is_dir():
    raise IOError(f'{str(destination)} is not a directory.')

# Get file paths
home_dir = Path('./test_data/Nutations')
nutation_files = list(home_dir.glob(f'{args.prefix}*.DTA'))
dsc_files = list(home_dir.glob(f'{args.prefix}*.DSC'))

nutations_folder = (destination / 'Nutations')
nutations_folder.mkdir()

nf = []
for dta, dsc in zip(nutation_files, dsc_files):
    nf.append(dta.rename(nutations_folder / dta.name))
    dsc.rename(nutations_folder / dsc.name)

cmap = plt.get_cmap('gray', len(nutation_files))
cmap = [rgb2hex(cmap(i)) for i in range(len(nutation_files))]

tau = 200  # ns
N = 2**14  # Zero Padding
ts, Vs = [], []
ffts, fft_freqs = [], []
freqs, nu = [], []
field = []
for i, file in enumerate(nf):
    t, V, params = dl.deerload(str(file), full_output=True)
    freqs.append(float(params['SPL']['MWFQ']) / 1e9)   # GHz
    field.append(int(float(params['SPL']['B0VL']) * 1e4))  # GHz
    V = dl.correctphase(V)
    if V.real[0] < 0:
        V = -V
    V -= V.mean()
    ts.append(t * 1e3), Vs.append(V)

    # Apply window and padding
    window = np.exp(-t / tau)
    nutation_win = window * V
    NWpad = np.zeros(N)
    NWpad[:len(nutation_win)] = nutation_win

    # Get FFT and frequency
    ft = np.fft.fftshift(np.fft.rfft(NWpad))
    dt = np.median(np.diff(t))
    f = np.fft.fftshift(np.fft.rfftfreq(N, dt))

    ffts.append(ft.real)
    fft_freqs.append(f.real)

    # Return FeqMax
    idxmax = np.argmax(ft.real)
    nu.append(f[idxmax])


output_file(str(destination / 'ResonatorProfile.html'))
source = ColumnDataSource(data=dict(freqs=freqs, nu=nu, field=field, ts=ts, Vs=Vs,
                                    colors=cmap, ffts=ffts, fft_freqs=fft_freqs))
hover = HoverTool(tooltips=[
                            ("Freq", "@freqs (GHz)"),
                            ("V", "@nu (MHz)"),
                            ("Field", "@field g"),
                            ])

nu = np.asarray(nu)
freqs = np.asarray(freqs)

line_plot = figure(plot_width=600, plot_height=300, x_axis_label='Time (ns)', tools=['reset'])
line_plot.multi_line(xs='ts', ys='Vs', line_color='colors', source=source, selection_color="orange")

fft_plot = figure(plot_width=600, plot_height=300, x_axis_label='Time (ns)', tools=['reset'])
fft_plot.multi_line(xs='fft_freqs', ys='ffts', line_color='colors', source=source, selection_color='orange')

scatter_plot = figure(plot_width=600, plot_height=300,
                      tools=[hover, 'tap,reset'],
                      x_axis_label='Frequency (GHz)',
                      y_axis_label='𝜈 (MHz)')

scatter_plot.circle(x='freqs', y='nu', size=10, source=source,  selection_color="orange")
scatter_plot.line(x='freqs', y='nu', source=source, nonselection_line_alpha=1)

show(column(line_plot, fft_plot, scatter_plot))

np.savetxt(str(destination / 'Transferfunction.dat'), np.asarray([freqs, nu]).T)




