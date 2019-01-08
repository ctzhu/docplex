'''
A command line interface / client for DOcplexcloud.

@author: kong
'''
from __future__ import print_function

import argparse
import datetime
import json
import os.path
from os.path import basename
import shlex
import sys

from six import iteritems

from docloud.job import JobClient
from docplex.mp.context import Context
from docloud.status import JobExecutionStatus


pd = None
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass

# This will be the ipython context if any
ip = None


try:
    from IPython.core.magic import Magics, magics_class, line_magic
    from IPython.core.display import display

    @magics_class
    class DocplexCliMagics(Magics):
        def __init__(self, shell):
            super(DocplexCliMagics, self).__init__(shell)
            self.url = os.environ.get('DOCPLEXCLOUD_URL')
            self.key = os.environ.get('DOCPLEXCLOUD_KEY')

        @line_magic
        def docplex_cli(self, line):
            "The docpelx CLI magics"
            args = shlex.split(line)
            try:
                return run_command('docplex_cli', args, url=self.url, key=self.key)
            except SystemExit:
                pass

        @line_magic
        def docplex_url(self, line):
            self.url = line
            return None

        @line_magic
        def docplex_key(self, line):
            self.key = line
            return None

    # register the magics
    try:
        ip = get_ipython()  # @UndefinedVariable
        ip.register_magics(DocplexCliMagics)
    except NameError:
        # get_ipython not found -> we are not in a notebook
        pass
except ImportError:
    # ipython is not available
    print("Could not import ipython things")
    pass


class ProgramResults(object):
    def __init__(self):
        self.return_code = 0
        self.output = []

    def __repr__(self):
        if self.return_code == 0:
            if self.output:
                return "\n".join(self.output)
            else:
                return ""
        else:
            return "Exit %s" % self.return_code

    def add_output(self, m):
        self.output.append(m)


class list_with_html_repr(list):
    def __init__(self):
        super(list_with_html_repr, self).__init__()

    def _repr_html_(self):
        st = "<ul>"
        for item in self:
            st += "<li>%s</lu" % item
        st += "</ul>"
        return st


def ls_jobs(client, program_result, quiet=False):
    jobs = client.get_all_jobs()
    if ip:
        result = []
    for i, j in enumerate(jobs):
        jobid = j["_id"]
        date = datetime.datetime.fromtimestamp(j['createdAt'] / 1e3)
        if ip:
            in_att = list_with_html_repr()
            out_att = list_with_html_repr()
            attachments = client.get_job_attachments(jobid)
            for a in attachments:
                desc = "%s (%s bytes)" % (a['name'], a['length'])
                if a['type'] == 'INPUT_ATTACHMENT':
                    in_att.append(desc)
                else:
                    out_att.append(desc)
            row = [jobid, j["executionStatus"], date, in_att, out_att]
            result.append(row)
        else:
            if not quiet:
                m = ("   [{0}] id={1} status={2} created={3}".format(i, jobid, j["executionStatus"],
                                                                                  date))
                attachments = client.get_job_attachments(jobid)
                for a in attachments:
                    m += ("\n      %s: %s (%s bytes)" % (a['type'], a['name'], a['length']))
            else:
                m = None
                print('%s' % jobid)
            if m:
                program_result.add_output(m)
    if ip:
        ar = np.array(result)
        result_df = pd.DataFrame(ar, index=range(len(jobs)), columns=['id', 'status', 'created', 'input attachments', 'output attachments'])
        with pd.option_context("display.max_colwidth", -1):
            display(result_df)


def rm_job(client, arguments, verbose=False):
    if len(arguments) == 1 and arguments[0] == 'all':
        arguments = [x["_id"] for x in client.get_all_jobs()]
    for id in arguments:
        try:
            if verbose:
                print("Deleting %s" % id)
            ok = client.delete_job(id)
            if not ok:
                print("Could not delete job %s" % id)
        except Exception as e:
            print(e)

last_updated_job = None


def print_job_info(info):
    global last_updated_job
    if 'details' in info:
        uptd = info.get('updatedAt', None)
        updatedated = datetime.datetime.fromtimestamp(float(uptd)/1000.0).strftime('%Y-%m-%d %H:%M:%S %Z') if uptd else None
        if updatedated != last_updated_job:
            msg = ('--- Solve details - %s ---' % updatedated)
            print(msg)
            last_updated_job = updatedated
            # let's format to align all ':'
            details = info['details']
            max_len = max([len(d) for d in details])
            output_format = '   %-' + str(max_len) + 's : %s'
            for k, v in iteritems(info['details']):
                print(output_format % (k, v))


def execute_job(client, inputs, verbose, details, nodelete):
    # The continuous logs feature is not available in version <= 1.0.257
    # we'll check TypeError when calling execute() and adjust parameters
    # if not available while waiting for the latest version to be out
    continuous_logs_available = False
    response = None
    try:
        if verbose:
            print("Executing")
        xkwargs = {'input': inputs,
                   'delete_on_completion': False}
        if details:
            xkwargs['info_cb'] = print_job_info
        try:
            xkwargs['continuous_logs'] = True
            try:
                response = client.execute(log=sys.stdout, **xkwargs)
                continuous_logs_available = True
            except TypeError as cla:
                if 'execute() got an unexpected keyword argument \'continuous_logs\'' in cla:
                    del xkwargs['continuous_logs']
                    response = client.execute(**xkwargs)
                else:
                    raise
        except TypeError as te:
            if 'execute() got an unexpected keyword argument \'info_cb\'' in te:
                print('Your version of docplexcloud client does not support details polling (--details option). Please update')
                sys.exit(-1)
            else:
                raise
        if response.execution_status != JobExecutionStatus.PROCESSED:
            print("Execution failed.\nDetails:\n%s" % json.dumps(response.job_info, sort_keys=True, indent=4))
        log_items = client.get_log_items(response.jobid)
        if not continuous_logs_available:
            # download and print logs if the continuous log feature was not available
            for log in log_items:
                for record in log["records"]:
                    print(record["message"])
        attachments = client.get_job_attachments(response.jobid)
        for a in attachments:
            if a['type'] == 'OUTPUT_ATTACHMENT':
                if verbose:
                    print("Downloading attachment %s" % a["name"])
                data = client.download_job_attachment(response.jobid, a["name"])
                with open(a["name"], "w+b") as f:
                    f.write(data)
    finally:
        if response and not nodelete:
            if verbose:
                print("Deleting job %s (cleanup)" % response.jobid)
            client.delete_job(response.jobid)


def run_command(prog, argv, url=None, key=None):
    description = '''Command line client for DOcplexcloud.'''
    epilog = '''Command details:
  info           Get and display information for the jobs which ids are
                 specified as ARG.
  download       Download the attachment to the the current directory.
  rm             Delete the jobs which ids are specfied as ARG.
  rm all         Delete all jobs.
  logs           Download and display the logs for the jobs which id are specified.
  ls             Lists the jobs.'''
    epilog_cli = '''
  execute        Submit a job and wait for end of execution. Each ARG that
                 is a file is uploaded as the job input. Example:
                    Example: python run.py execute model.py model.data -v
                      executes a job which input files are model.py and
                      model.dada, in verbose mode.
'''

    if ip is None:
        epilog += epilog_cli
    parser = argparse.ArgumentParser(prog=prog, description=description, epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command',
                        metavar='COMMAND',
                        help='DOcplexcloud command')
    parser.add_argument('arguments', metavar='ARG', nargs='*',
                        help='Arguments for the command')
    parser.add_argument('--no-delete', action='store_true', default=False,
                        dest='nodelete',
                        help="If specified, jobs are not deleted after execution")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose mode')
    parser.add_argument('--as', nargs=1, metavar='HOST',
                        dest="host_config", default=None,
                        help="'as host' - use the cplex_config_<HOST>.py configuration file found in PYTHONPATH")
    parser.add_argument('--url', nargs=1, metavar='URL',
                        dest="url", default=None,
                        help="The DOcplexcloud connection URL. If not specified, will use those found in docplex config files")
    parser.add_argument('--key', nargs=1, metavar='API_KEY',
                        dest="key", default=None,
                        help="The DOcplexcloud connection key. If not specified, will use those found in docplex config files")
    parser.add_argument('--details', action='store_true', default=False,
                        help='Display solve details as they are available')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Only show numeric IDs as output')
    args = parser.parse_args(argv)

    program_result = ProgramResults()

    # Get the context here so that we have some credentials at hand
    context = Context.make_default_context()

    if args.host_config is not None:
        config_name = "cplex_config_%s.py" % args.host_config[0]
        config_file = list(filter(os.path.isfile, [os.path.join(x, config_name) for x in sys.path]))
        if len(config_file) == 0:
            print("Could not find config file for host: %s" % args.host_config[0])
            program_result.return_code = -1
            return(program_result)
        if args.verbose:
            print("Overriding host config with: %s" % config_file[0])
        context.read_settings(config_file[0])

    # use credentials in context unless they are given to this function
    client_url = context.solver.docloud.url if url is None else url
    client_key = context.solver.docloud.key if key is None else key
    # but if there are some credentials in arguments (--url, --key), use them
    if args.url:
        client_url = args.url
    if args.key:
        client_key = args.key
    if args.verbose:
        print('**** Connecting to %s with key %s' % (client_url, client_key))
        print('Will send command %s' % args.command)
        print('Arguments:')
        for i in args.arguments:
            print('  -> %s' % i)
        print('verbose = %s' % args.verbose)

    client = JobClient(client_url, client_key)

    if args.command == 'ls':
        ls_jobs(client, program_result, quiet=args.quiet)
    elif args.command == 'info':
        if len(args.arguments) == 1 and args.arguments[0] == 'all':
            args.arguments = [x["_id"] for x in client.get_all_jobs()]
        for id in args.arguments:
            info_text = "NOT FOUND"
            try:
                job = client.get_job(id)
                info_text = json.dumps(job, indent=3)
            except:
                pass
            print("%s:\n%s" % (id, info_text))
    elif args.command == 'rm':
        if args.arguments:
            joblist = args.arguments
        else:
            joblist = shlex.split(sys.stdin.read())
        rm_job(client, joblist, verbose=args.verbose)
    elif args.command == 'logs':
        for jid in args.arguments:
            log_items = client.get_log_items(jid)
            for log in log_items:
                for record in log["records"]:
                    print(record["message"])
    elif args.command == 'download':
        for jid in args.arguments:
            job = client.get_job(jid)
            for attachment in job['attachments']:
                print('downloading %s' % attachment['name'])
                with open(attachment['name'], 'wb') as f:
                    f.write(client.download_job_attachment(id, attachment['name']))
    elif args.command == 'execute':
        inputs = [{'name': basename(a), 'filename': a} for a in args.arguments]
        if args.verbose:
            for i in inputs:
                print("Uploading %s as attachment name %s" % (i['filename'], i['name']))
        execute_job(client, inputs, args.verbose, args.details, args.nodelete)
    else:
        print("Unknown command: %s" % args.command)
        program_result.return_code = -1
        return(program_result)
    return(program_result)


if __name__ == '__main__':
    program_result = run_command(sys.argv[0], sys.argv[1:])
    if program_result.output:
        exit(program_result)
    else:
        exit(program_result.return_code)
