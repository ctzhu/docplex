from distutils.core import setup
required=[ 'requests',
           'six',
           'docloud>=1.0.132']
import sys
if ((sys.version_info[0]) < 3) or ((sys.version_info[0] == 3) and (sys.version_info[1] < 2)):
    required.append('futures')


import os
import re
HERE = os.path.abspath(os.path.dirname(__file__))
def read(*parts):
    try:
        with open(os.path.join(HERE, *parts)) as f:
            return f.read()
    except:
        return "docplex"
ss = str(read('README.rst'))

setup(
    name = 'docplex',
    packages = ['docplex',
                'docplex.cp',
                'docplex.cp.solver',
                'docplex.mp',
                'docplex.mp.internal',
                'docplex.mp.params',
                'docplex.mp.worker',
                'docplex.util'],
    version = '1.0.607',  # replaced at build time
    description = 'The IBM Decision Optimization CPLEX Modeling for Python',
    author = 'The IBM Decision Optimization on Cloud team',
    author_email = 'dofeedback@wwpdl.vnet.ibm.com',
    long_description='%s\n' % ss,
    url = 'https://onboarding-oaas.docloud.ibmcloud.com/software/analytics/docloud/',
    keywords = ['docloud', 'optimization', 'cplex', 'cpo'],
    license = read('LICENSE.txt'),
    install_requires=required,
    classifiers = ["Development Status :: 5 - Production/Stable",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Information Technology",
                   "Intended Audience :: Science/Research",
                   "Operating System :: Unix",
                   "Operating System :: MacOS",
                   "Operating System :: Microsoft",
                   "Operating System :: OS Independent",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: Software Development :: Libraries",
                   "Topic :: System",
                   "Topic :: Other/Nonlisted Topic",
                   "License :: OSI Approved :: Apache Software License",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3.4"
                   ],
)

print("** The documentation can be found here: https://github.com/IBMDecisionOptimization/docplex-doc")
print("** The examples can be found here: https://github.com/IBMDecisionOptimization/docplex-examples")
