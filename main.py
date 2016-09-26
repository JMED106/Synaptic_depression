import argparse
import yaml
import sys
from timeit import default_timer as timer
import progressbar as pb
import Gnuplot

import numpy as np
from frlib import Data, FiringRate
from tools import qifint, qifint_noise, TheoreticalComputations, SaveResults, Perturbation, noise

__author__ = 'jm'


# Empty class to manage external parameters
# noinspection PyClassHasNoInit
class Options:
    pass


options = None
ops = Options()
pi = np.pi
pi2 = np.pi * np.pi

# We first try to parse optional configuration files:
fparser = argparse.ArgumentParser(add_help=False)
fparser.add_argument('-f', '--file', default="conf.txt", dest='-f', metavar='<file>')
farg = fparser.parse_known_args()
conffile = vars(farg[0])['-f']

# We open the configuration file to load parameters (not optional)
try:
    options = yaml.load(file(conffile, 'rstored'))
except IOError:
    print "The configuration file '%s' is missing" % conffile
    exit(-1)
except yaml.YAMLError, exc:
    print "Error in configuration file:", exc
    exit(-1)

# We load parameters from the dictionary of the conf file and add command line options (2nd parsing)
parser = argparse.ArgumentParser(
    description='Simulator of an all-to-all network of QIF neurons with synaptic depression dynamics.',
    usage='python %s [-O <options>]' % sys.argv[0])

for group in options:
    gr = parser.add_argument_group(group)
    for key in options[group]:
        flags = key.split()
        opts = options[group]
        gr.add_argument(*flags, default=opts[key]['default'], help=opts[key]['description'], dest=flags[0][1:],
                        metavar=opts[key]['name'], type=type(opts[key]['default']),
                        choices=opts[key]['choices'])

# We parse command line arguments:
opts = parser.parse_args(farg[1])
args = parser.parse_args(farg[1], namespace=ops)

###################################################################################
# 0) PREPARE FOR CALCULATIONS
# 0.1) Load data object:
d = Data(n=args.N, eta0=args.e, delta=args.d, tfinal=args.T, dt=float(args.dt), faketau=args.tm, taud=args.td, u=args.U,
         fp=args.D, system=args.s, j0=args.j)

# 0.2) Load initial conditions
if args.oic is False:
    d.load_ic(args.j, system=d.system)
else:
    d.fileprm = '%s_%.2lf-%.2lf-%.2lf' % (d.fp, d.j0, d.eta0, d.delta)


if args.ic:
    print "Forcing initial conditions generation..."
    d.new_ic = True

if d.new_ic:
    d.u = 0.0
    d.d[-1] = 0.9
    d.dqif[-1] = 0.9

# 0.3) Load Firing rate class in case qif network is simulated
if d.system != 'fr':
    fr = FiringRate(data=d, swindow=0.1, sampling=0.05)

# 0.4) Set perturbation configuration
p = Perturbation(data=d, dt=args.pt, amplitude=args.a, attack=args.A)

# 0.5) Define saving paths:
sr = SaveResults(data=d, pert=p, system=d.system, parameters=opts)

# 0.6) Other theoretical tools:
th = TheoreticalComputations(d, p)

# Progress-bar configuration
widgets = ['Progress: ', pb.Percentage(), ' ',
           pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA(), ' ']

###################################################################################
# 1) Simulation (Integrate the system)
print('Simulating ...')
pbar = pb.ProgressBar(widgets=widgets, maxval=10 * (d.nsteps + 1)).start()
time1 = timer()
tstep = 0
temps = 0

# Time loop
while temps < d.tfinal:
    # Time step variables
    kp = tstep % d.nsteps
    k = (tstep + d.nsteps - 1) % d.nsteps
    # ######################## - PERTURBATION  - ##
    if p.pbool and not d.new_ic:
        if temps >= p.t0:
            p.timeevo(temps)
    p.it[kp] = p.input

    # ######################## -  INTEGRATION  - ##
    # ######################## -      qif      - ##
    if d.system == 'qif' or d.system == 'both':
        tsyp = tstep % d.T_syn
        tskp = tstep % d.spiketime
        tsk = (tstep + d.spiketime - 1) % d.spiketime
        # We compute the Mean-field vector s_j
        # noinspection PyUnresolvedReferences
        s = (1.0 / d.N) * np.add.reduce(np.dot(d.spikes, d.a_tau[:, tsyp]))
        d.dqif[kp] = d.dqif[k] + d.dt * ((1.0 - d.dqif[k]) / d.taud - d.u * s * d.dqif[k])

        if d.fp == 'noise':
            noiseinput = np.sqrt(2.0 * d.dt / d.tau * d.delta) * noise(d.N)
            # Excitatory
            d.matrix = qifint_noise(d.matrix, d.matrix[:, 0], d.matrix[:, 1], d.eta0, d.j0 * d.dqif[kp] * s + p.input,
                                    noiseinput, temps, d.N, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)
        else:
            # Excitatory
            d.matrix = qifint(d.matrix, d.matrix[:, 0], d.matrix[:, 1], d.eta, d.j0 * d.dqif[kp] * s + p.input, temps,
                              d.N, d.dt, d.tau, d.vpeak, d.refr_tau, d.tau_peak)

        # Prepare spike matrices for Mean-Field computation and firing rate measure
        # Excitatory
        d.spikes_mod[:, tsk] = 1 * d.matrix[:, 2]  # We store the spikes
        d.spikes[:, tsyp] = 1 * d.spikes_mod[:, tskp]

        # If we are just obtaining the initial conditions (a steady state) we don't need to
        # compute the firing rate.
        if not d.new_ic:
            # Voltage measure:
            vma = (d.matrix[:, 1] <= temps)  # Neurons which are not in the refractory period
            fr.vavg0[vma] += d.matrix[vma, 0]
            fr.vavg += 1

            # ######################## -- FIRING RATE MEASURE -- ##
            fr.frspikes[:, tstep % fr.wsteps] = 1 * d.spikes[:, tsyp]
            fr.firingrate(tstep)
            # Distribution of Firing Rates
            if tstep > 0:
                fr.tspikes2 += d.matrix[:, 2]
                fr.ravg2 += 1  # Counter for the "instantaneous" distribution
                fr.ravg += 1  # Counter for the "total time average" distribution

    # ######################## -  INTEGRATION  - ##
    # ######################## --   FR EQS.   -- ##
    if d.system == 'fr' or d.system == 'both':
        # -- Integration -- #
        d.d[kp] = d.d[k] + d.dt * ((1.0 - d.d[k]) / d.taud - d.u * d.r[k] * d.d[k])
        d.r[kp] = d.r[k] + d.dt * (d.delta / pi + 2.0 * d.r[k] * d.v[k])
        d.v[kp] = d.v[k] + d.dt * (d.v[k] * d.v[k] + d.eta0 - pi2 * d.r[k] * d.r[k] + d.j0 * d.r[k] * d.d[kp] + p.input)
    # Perturbation at certain time
    if int(p.t0 / d.dt) == tstep:
        p.pbool = True

    # Time evolution
    pbar.update(10 * tstep + 1)
    temps += d.dt
    tstep += 1

# Finish pbar
pbar.finish()
# Stop the timer
print 'Total time: {}.'.format(timer() - time1)

# Compute distribution of firing rates of neurons
tstep -= 1
temps -= d.dt
th.thdist = th.theor_distrb(d.d[tstep % d.nsteps], 1)

# Save initial conditions
if d.new_ic:
    d.save_ic(temps)
    exit(0)

# Register data to a dictionary
if 'qif' in d.systems:
    # Distribution of firing rates over all time
    fr.frqif0 = fr.tspikes / (fr.ravg * d.dt) / d.faketau

    if 'fr' in d.systems:
        d.register_ts(fr, th)
    else:
        d.register_ts(fr)
else:
    d.register_ts(th=th)

# Save results
sr.create_dict()
sr.results['perturbation']['It'] = p.it
sr.save()

# Preliminar plotting with gnuplot
gp = Gnuplot.Gnuplot(persist=1)
p1 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, d.r / d.faketau], with_='lines')
if opts.s != 'fr':
    p2 = Gnuplot.PlotItems.Data(np.c_[np.array(fr.tempsfr) * d.faketau, np.array(fr.r) / d.faketau],
                                with_='lines')
else:
    p2 = Gnuplot.PlotItems.Data(np.c_[d.tpoints * d.faketau, d.d / d.faketau], with_='lines')
gp.plot(p1, p2)
