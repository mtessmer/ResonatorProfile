import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import deerlab as dl




ansx, ansy = np.loadtxt('test_data/Transferfunction.dat').T
nutation_files = glob('test_data/Nutations/Nut_*.DTA')
cmap = plt.get_cmap('twilight', len(nutation_files))

tau = 200  # ns
N = 2**14  # Zero Padding
fig, (ax1, ax2) = plt.subplots(2)

freqs = []
powers = []
for i, file in enumerate(nutation_files):
    t, V, params = dl.deerload(file, full_output=True)
    freqs.append(float(params['SPL']['MWFQ']) / 1e9)   # GHz
    V = dl.correctphase(V)
    V -= V.mean()
    window = np.exp(-t / tau)
    nutation_win = window * V
    NWpad = np.zeros(N)
    NWpad[:len(nutation_win)] = nutation_win

    ft = np.fft.fftshift(np.fft.rfft(NWpad))
    dt = np.median(np.diff(t))
    f = np.fft.fftshift(np.fft.rfftfreq(N, dt))

    idxmax = np.argmax(np.abs(ft))
    powers.append(f[idxmax])
    ax1.plot(t, V, color=cmap(i))

powers = np.asarray(powers)
freqs = np.asarray(freqs)
ax2.plot(freqs, powers)
ax2.scatter(freqs, powers, edgecolor='k', facecolor='none')
ax2.set_ylabel('Frequency (MHz)')
ax2.set_ylabel('Frequency (GHz)')

plt.show()




