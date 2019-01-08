# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model expressed as a CPO file using the
CPO solver 'angel' executable (written in C++)
"""

from docplex.cp.solution import *
from docplex.cp.utils import CpoException, StringIO
import docplex.cp.solver.solver as solver

import subprocess
import sys
import time
import threading
import json


###############################################################################
##  Private constants
###############################################################################

# List of command ids that can be sent to solver
CMD_EXIT            = "Exit"           # End process (no data)
CMD_SET_CPO_MODEL   = "SetCpoModel"    # CPO model as string
CMD_SOLVE_MODEL     = "SolveModel"     # Complete solve of the model (no data)
CMD_START_SEARCH    = "StartSearch"    # Start search (no data)
CMD_SEARCH_NEXT     = "SearchNext"     # Get next solution (no data)
CMD_END_SEARCH      = "EndSearch"      # End search (no data)
CMD_REFINE_CONFLICT = "RefineConflict" # Refine conflict (no data)
CMD_PROPAGATE       = "Propagate"      # Propagate (no data)

# List of events received from solver
EVT_VERSION_INFO       = "VersionInfo"     # Angel version info (String in JSON format)
EVT_SUCCESS            = "Success"         # Success in last command execution
EVT_ERROR              = "Error"           # Error (data is error string)
EVT_TRACE              = "DebugTrace"      # Debugging trace
EVT_SOLVER_OUT_STREAM  = "OutStream"       # Solver output stream
EVT_SOLVER_WARN_STREAM = "WarningStream"   # Solver warning stream
EVT_SOLVER_ERR_STREAM  = "ErrorStream"     # Solver error stream
EVT_SOLVE_RESULT       = "SolveResult"     # Solver result in JSON format
EVT_CONFLICT_RESULT    = "ConflictResult"  # Conflict refiner result in JSON format
EVT_PROPAGATE_RESULT   = "PropagateResult" # Propagate result in JSON format

# Max possible received data size in one message
_MAX_RECEIVED_DATA_SIZE = 1000000

# Python 3 indicator
IS_PYTHON_2 = (sys.version_info[0] == 2)


###############################################################################
##  Public classes
###############################################################################

class AngelException(CpoException):
    """ The base class for exceptions raised by the Angel client
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(AngelException, self).__init__(msg)



class CpoSolverAngel(solver.CpoSolverAgent):
    """ Interface to a local solver through an external process """

    def __init__(self, model, params, context):
        """ Create a new solver using local proxy.

        Args:
            model:    Model to solve
            params:   Solving parameters
            context:  Proxy solver context
        Raises:
            CpoException if proxy executable does not exists
        """
        # Call super
        super(CpoSolverAngel, self).__init__(model, params, context)
        self.process = None

        # Check if executable file exists
        if context.execfile is None:
            raise CpoException("Executable file should be given in 'execfile' context attribute.")
        #if not os.path.isfile(context.execfile):
        #    raise CpoException("Executable file '" + str(context.execfile) + "' does not exists")

        # Init log elements
        self.lout = context.get_log_output()
        self.printlog = context.trace_log and (self.lout is not None)
        self.loglines = [] if context.add_log_to_solution else None
        self.logenabled = self.printlog or (self.loglines is not None)

        # Create solving process
        cmd = [context.execfile]
        if context.parameters is not None:
            cmd.extend(context.parameters)
        context.log(2, "Angel exec command: '", ' '.join(cmd), "'")
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, universal_newlines=False)
        except:
            raise CpoException("Can not execute command '{}'. Please check availability of required executable file.".format(' '.join(cmd)))
        self.pout = self.process.stdin
        self.pin = self.process.stdout

        # Read initial version info from process
        self.version = None
        timer = threading.Timer(1, lambda: self.process.kill() if self.version is None else None)
        timer.start()
        evt, data = self._read_message()
        timer.cancel()
        if evt != EVT_VERSION_INFO:
            raise AngelException("Unexpected event {} received instead of version number event {}.".format(evt, EVT_VERSION_INFO))
        self.version = json.loads(data.decode('utf-8'))
        context.log(3, "Angel version: '", self.version, "'")

        # Convert model into CPO format
        cpostr = self._get_cpo_model_string()

        # Send CPO model to process
        self._write_message(CMD_SET_CPO_MODEL, cpostr)
        context.log(3, "Model sent, wait for solution")
        self._wait_event(EVT_SUCCESS)


    def __del__(self):
        # End solve
        self.end()


    def solve(self):
        """ Solve the model

        According to the value of the context parameter 'verbose', the following information is logged
        if the log output is set:
         * 1: Total time spent to solve the model
         * 2: The process exec file
         * 3: Content of the JSON response
         * 4: Solver traces (if any)
         * 5: Messages sent/receive to/from process

        Returns:
            Model solve result (object of class CpoSolveResult)
        """

        # Start solve
        self._write_message(CMD_SOLVE_MODEL)

        # Wait for next solution
        return self._wait_json_result()


    def start_search(self):
        """ Start a new search. Solutions are retrieved using method search_next().
        """
        self._write_message(CMD_START_SEARCH)


    def search_next(self):
        """ Get the next available solution.

        (This method starts search automatically.)

        Returns:
            Next model result (type CpoSolveResult)
        """

        # Request next solution
        self._write_message(CMD_SEARCH_NEXT)

        # Wait next solution
        return self._wait_json_result()


    def end_search(self):
        """ End current search.

        Returns:
            Last (fail) model solution with last solve information (type CpoSolveResult)
        """

        # Request end search
        self._write_message(CMD_END_SEARCH)

        # Wait next solution
        return self._wait_json_result()


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        See documentation of CpoSolver.refine_conflict() for details.

        Returns:
            Conflict result (object of class CpoRefineConflictResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        # Ensure cpo model string has been generated with all constraints named
        if not self.model._ensure_all_root_constraints_named():
            # Build and send new CPO model string
            self.context.model.name_all_constraints = True
            cpostr = self._get_cpo_model_string()
            self._write_message(CMD_SET_CPO_MODEL, cpostr)
            self._wait_event(EVT_SUCCESS)

        # Request refine conflict
        self._write_message(CMD_REFINE_CONFLICT)

        # Wait next solution
        return self._wait_json_conflict()


    def propagate(self):
        """ This method invokes the propagation on the current model.

        See documentation of CpoSolver.propagate() for details.

        Returns:
            Conflict result (object of class CpoRefineConflictResult)
        Raises:
            CpoNotSupportedException: method not available in this solver agent.
        """
        # Request propagation
        self._write_message(CMD_PROPAGATE)

        # Wait next solution
        return self._wait_json_propagate()


    def end(self):
        """ End solver and release all resources.
        """
        if self.process is not None:
            try:
                self._write_message(CMD_EXIT)
            except:
                pass
            time.sleep(0.300)
            self.process.kill()
            self.process = None
            super(CpoSolverAngel, self).end()


    def _wait_event(self, xevt):
        """ Wait for a particular event while forwarding logs if any.
        Args:
            xevt: Expected event
        Returns:
            Message data
        Raises:
            AngelException if an error occurs
        """
        # Initialize first error string to enrich exception if any
        firsterror = None

        # Read events
        while True:
            # Read and process next message
            evt, data = self._read_message()
            if evt == xevt:
                return data
            elif evt in (EVT_SOLVER_OUT_STREAM, EVT_SOLVER_WARN_STREAM):
                if self.logenabled:
                    ldata = data.decode('utf-8')
                    if self.printlog:
                        self.lout.write(ldata)
                        self.lout.flush()
                    if self.loglines is not None:
                        self.loglines.append(ldata)
            elif evt == EVT_SOLVER_ERR_STREAM:
                ldata = data.decode('utf-8')
                if firsterror is None:
                    firsterror = ldata.replace('\n', '')
                out = self.lout if self.lout is not None else sys.stdout
                out.write("ERROR: ")
                out.write(ldata)
                out.flush()
            elif evt == EVT_TRACE:
                self.context.log(4, data.decode('utf-8'))
            elif evt == EVT_ERROR:
                errmsg = data.decode('utf-8')
                if firsterror is not None:
                    errmsg += " (" + firsterror + ")"
                self.end()
                raise AngelException(errmsg)
            else:
                self.end()
                raise AngelException("Unknown event received from solver angel: " + str(evt))

        # Return
        return data


    def _wait_json_result(self):
        """ Wait for a solution while forwarding logs if any.
        Returns:
            Model solution (type CpoModelSolution)
        """

        # Wait JSON result
        data = self._wait_event(EVT_SOLVE_RESULT)
        jsol = data.decode('utf-8')
        self._set_last_json_result_string(jsol)

        # Build response solution
        msol = CpoSolveResult(self.model)
        msol._add_json_solution(jsol)
        if self.loglines is not None:
            msol._set_solver_log(''.join(self.loglines))
            self.loglines = []

        return msol


    def _wait_json_conflict(self):
        """ Wait for a conflict while forwarding logs if any.
        Returns:
            Conflict result (type CpoConflictRefinerResult)
        """

        # Wait JSON result
        data = self._wait_event(EVT_CONFLICT_RESULT)
        jsol = data.decode('utf-8')
        self._set_last_json_result_string(jsol)

        # Build response
        csol = CpoRefineConflictResult(self.model)
        csol._add_json_solution(jsol)
        if self.loglines is not None:
            csol._set_solver_log(''.join(self.loglines))
            self.loglines = []

        return csol


    def _wait_json_propagate(self):
        """ Wait for a solution while forwarding logs if any.
        Returns:
            Partial model solution (type CpoModelSolution)
        """

        # Wait JSON result
        data = self._wait_event(EVT_PROPAGATE_RESULT)
        jsol = data.decode('utf-8')
        self._set_last_json_result_string(jsol)

        # Build response solution
        msol = CpoSolveResult(self.model)
        msol._add_json_solution(jsol)
        if self.loglines is not None:
            msol._set_solver_log(''.join(self.loglines))
            self.loglines = []

        return msol


    def _write_message(self, cid, data=None):
        """ Write a message to the solver process
        Args:
            cid:   Command name
            data:  Value to write
        """
        # Build header
        cid = cid.encode('utf-8')
        tlen = len(cid)
        if data is not None:
            data = data.encode('utf-8')
            tlen += len(data) + 1
        if tlen > 0xffffffff:
            raise AngelException("Try to send a message with length {}, greater than {}.".format(tlen, 0xffffffff))
        frame = bytearray(6)
        frame[0] = 0xCA
        frame[1] = 0xFE
        frame[2] = (tlen >> 24) & 0xFF
        frame[3] = (tlen >> 16) & 0xFF
        frame[4] = (tlen >> 8)  & 0xFF
        frame[5] = tlen         & 0xFF

        # Add data if any
        if data is None:
            frame = frame + cid
        else:
            frame = frame + cid + bytearray(1) + data
        self.context.log(5, "Send message: cmd=", cid, ", tsize=", tlen)

        # Write message frame
        self.pout.write(frame)
        self.pout.flush()


    def _read_message(self):
        """ Read a message from the solver process
        Returns:
            Tuple (evt, data)
        """
        # Read message header
        frame = self._read_frame(6)
        if (frame[0] != 0xCA) or (frame[1] != 0xFE):
            self.end()
            raise AngelException("Invalid message header '{}'. Probable wrong destination or desynchronization of stream.".format(frame))

        # Read message data
        tsize = (frame[2] << 24) | (frame[3] << 16) | (frame[4] << 8) | frame[5]
        data = self._read_frame(tsize)

        # Split name from data
        ename = 0
        while (ename < tsize) and (data[ename] != 0):
            ename += 1
        if ename == tsize:
            # Command only, no data
            evt = data.decode('utf-8')
            data = None
        else:
            # Split command and data
            evt = data[0:ename].decode('utf-8')
            data = data[ename+1:]
        self.context.log(5, "Read message: ", evt, ", data: '", data, "'")
        return evt, data


    def _read_frame(self, nbb):
        """ Read a byte frame from input stream
        Args:
            nbb:  Number of bytes to read
        Returns:
            Byte array
        """
        # Read data
        data = self.pin.read(nbb)
        if len(data) != nbb:
            if len(data) == 0:
                raise AngelException("Nothing to read from angel process.")
            else:
                raise AngelException("Read only {} bytes when {} was expected.".format(len(data), nbb))
        # Return
        if IS_PYTHON_2:
            return bytearray(data)
        return data


###############################################################################
##  Private Functions
###############################################################################



