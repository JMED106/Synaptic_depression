import argparse
import yaml
import sys

__author__ = 'jm'


# Empty class to manage external parameters
# noinspection PyClassHasNoInit
class Options:
    pass


options = None
ops = Options()

# We first try to parse optional configuration files:
fparser = argparse.ArgumentParser(add_help=False)
fparser.add_argument('-f', '--file', default="conf2.txt", dest='-f', metavar='<file>')
farg = fparser.parse_known_args()
conffile = vars(farg[0])['-f']

# We open the configuration file to load parameters (not optional)
try:
    options = yaml.load(file(conffile, 'r'))
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
args = parser.parse_args(farg[1], namespace=ops)

extopts = {"dt": 1E-3, "t0": 0.0, "ftau": 20.0E-3, "modes": [10, 7.5, -2.5]}
pertopts = {"dt": 0.5, "attack": 'exponential', "release": 'instantaneous'}
