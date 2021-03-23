import argparse
from glob import glob
import numpy as np
from matplotlib.colors import rgb2hex
import matplotlib.pyplot as plt
from bokeh.models import HoverTool, Select
from bokeh.layouts import column
from bokeh.plotting import figure, output_file, show, ColumnDataSource
import deerlab as dl

nutation_files = glob('test_data/Nutations/Nut_*.DTA')
cmap = plt.get_cmap('gray', len(nutation_files))
cmap = [rgb2hex(cmap(i)) for i in range(len(nutation_files))]

tau = 200  # ns
N = 2**14  # Zero Padding
ts, Vs = [], []
freqs, powers = [], []
field = []
for i, file in enumerate(nutation_files):
    t, V, params = dl.deerload(file, full_output=True)
    freqs.append(float(params['SPL']['MWFQ']) / 1e9)   # GHz
    field.append(int(float(params['SPL']['B0VL']) * 1e4))  # GHz
    V = dl.correctphase(V)
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

    # Return FeqMax
    idxmax = np.argmax(np.abs(ft))
    powers.append(f[idxmax])



source = ColumnDataSource(data=dict(freqs=freqs, powers=powers, field=field, ts=ts, Vs=Vs, colors=cmap))
hover = HoverTool(tooltips=[
                            ("Freq", "@freqs (GHz)"),
                            ("V", "@powers (MHz)"),
                            ("Field", "@field g"),
                            ])

powers = np.asarray(powers)
freqs = np.asarray(freqs)




line_plot = figure(plot_width=600, plot_height=300, x_axis_label='Time (ns)', tools=['reset'])
line_plot.multi_line(xs='ts', ys='Vs', line_color='colors', source=source, selection_color="orange")

scatter_plot = figure(plot_width=600, plot_height=300,
                      tools=[hover, 'tap,reset'],
                      x_axis_label='Frequency (GHz)',
                      y_axis_label='V (MHz)')

scatter_plot.circle(x='freqs', y='powers', size=10, source=source,  selection_color="orange")
scatter_plot.line(x='freqs', y='powers', source=source, nonselection_line_alpha=1)


show(column(line_plot, scatter_plot))





