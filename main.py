import getopt
import sys

import yaml

__author__ = 'jm'


def main(argv, options):
    try:
        optis, args = getopt.getopt(argv, "hm:a:s:c:N:n:e:d:t:D:f:",
                                    ["mode=", "amp=", "system=", "connec=", "neurons=", "lenght=", "extcurr=",
                                     "delta=", "tfinal=", "Distr=", "file="])
    except getopt.GetoptError:
        print 'main.py [-m <mode> -a <amplitude> -s <system> -c <connectivity> ' \
              '-N <number-of-neurons> -n <lenght-of-ring-e <external-current> ' \
              '-d <widt-of-dist> -t <final-t> -D <type-of-distr> -f <config-file>]'
        sys.exit(2)

    for opt, arg in optis:
        if len(opt) > 2:
            opt = opt[1:3]
        opt = opt[1]
        # Check type and cast
        if isinstance(options[opt], int):
            options[opt] = int(float(arg))
        elif isinstance(options[opt], float):
            options[opt] = float(arg)
        else:
            options[opt] = arg

    return options


opts = {"m": 0, "a": 1.0, "s": 'both', "c": 'mex-hat',
        "N": int(2E5), "n": 100, "e": 4.0, "d": 0.5, "t": 20,
        "D": 'lorentz', "f": "conf.txt"}
extopts = {"dt": 1E-3, "t0": 0.0, "ftau": 20.0E-3, "modes": [10, 7.5, -2.5]}
pertopts = {"dt": 0.5, "attack": 'exponential', "release": 'instantaneous'}

if __name__ == '__main__':
    opts2 = main(sys.argv[1:], opts)
else:
    opts2 = opts
try:
    (opts, extopts, pertopts) = yaml.load(file(opts2['f']))
    if __name__ == '__main__':
        opts = main(sys.argv[1:], opts)
except IOError:
    print "The configuration file %s is missing, using inbuilt configuration." % (opts2['f'])
except ValueError:
    print "Configuration file has bad format."
    exit(-1)
