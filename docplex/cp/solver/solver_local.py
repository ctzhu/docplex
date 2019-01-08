# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------
# Author: Olivier OUDOT, IBM Analytics, France Lab, Sophia-Antipolis

"""
This module allows to solve a model expressed as a CPO file using
a local CP Optimizer Interactive (cpoptimizer(.exe)).
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

class LocalSolverException(CpoException):
    """ The base class for exceptions raised by the local solver client
    """
    def __init__(self, msg):
        """ Create a new exception
        Args:
            msg: Error message
        """
        super(LocalSolverException, self).__init__(msg)


class CpoSolverLocal(solver.CpoSolverAgent):
    """ Interface to a local solver through an external process """

    def __init__(self, model, params, context):
        """ Create a new solver that solves locally with CP Optimizer Interactive.

        Args:
            model:    Model to solve
            params:   Solving parameters
            context:  Proxy solver context
        Raises:
            CpoException if proxy executable does not exists
        """
        # Call super
        super(CpoSolverLocal, self).__init__(model, params, context)
        self.process = None

        # Check if executable file exists
        if context.execfile is None:
            raise CpoException("Executable file should be given in 'execfile' context attribute.")
        if not is_string(context.execfile):
            raise CpoException("Executable file should be given in 'execfile' as a string.")
        #if not os.path.isfile(context.execfile):
        #    raise CpoException("Executable file '" + str(context.execfile) + "' does not exists")

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
            raise LocalSolverException("Unexpected event {} received instead of version number event {}.".format(evt, EVT_VERSION_INFO))
        self.version = json.loads(data.decode('utf-8'))
        context.log(3, "Angel version: '", self.version, "'")

        # Convert model into CPO format
        cpostr = self._get_cpo_model_string()

        # Encode model
        stime = time.time()
        cpostr = cpostr.encode('utf-8')
        self.process_infos[CpoProcessInfos.MODEL_ENCODE_TIME] = time.time() - stime

        # Send CPO model to process
        stime = time.time()
        self._write_message(CMD_SET_CPO_MODEL, cpostr)
        self.process_infos[CpoProcessInfos.MODEL_SEND_TIME] = time.time() - stime
        context.log(3, "Model sent.")
        self._wait_json_result(EVT_SUCCESS)  # JSON stored


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
            Model solve result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """

        # Start solve
        self._write_message(CMD_SOLVE_MODEL)

        # Wait JSON result
        jsol = self._wait_json_result(EVT_SOLVE_RESULT)

        # Build result object
        return self._create_result_object(CpoSolveResult, jsol)


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

        # Wait JSON result
        jsol = self._wait_json_result(EVT_SOLVE_RESULT)

        # Build result object
        return self._create_result_object(CpoSolveResult, jsol)


    def end_search(self):
        """ End current search.

        Returns:
            Last (fail) model solution with last solve information (type CpoSolveResult)
        """

        # Request end search
        self._write_message(CMD_END_SEARCH)

        # Wait JSON result
        jsol = self._wait_json_result(EVT_SOLVE_RESULT)

        # Build result object
        return self._create_result_object(CpoSolveResult, jsol)


    def refine_conflict(self):
        """ This method identifies a minimal conflict for the infeasibility of the current model.

        See documentation of CpoSolver.refine_conflict() for details.

        Returns:
            Conflict result,
            object of class :class:`~docplex.cp.solution.CpoRefineConflictResult`.
        """
        # Ensure cpo model string has been generated with all constraints named
        if not self.model._ensure_all_root_constraints_named():
            # Build and send new CPO model string
            self.context.model.name_all_constraints = True
            cpostr = self._get_cpo_model_string()
            # Encode model
            stime = time.time()
            cpostr = cpostr.encode('utf-8')
            self.process_infos.incr(CpoProcessInfos.MODEL_ENCODE_TIME, time.time() - stime)
            # Send CPO model to process
            stime = time.time()
            self._write_message(CMD_SET_CPO_MODEL, cpostr)
            self.process_infos.incr(CpoProcessInfos.MODEL_SEND_TIME, time.time() - stime)

            self._wait_event(EVT_SUCCESS)

        # Request refine conflict
        self._write_message(CMD_REFINE_CONFLICT)

        # Wait JSON result
        jsol = self._wait_json_result(EVT_CONFLICT_RESULT)

        # Build result object
        return self._create_result_object(CpoRefineConflictResult, jsol)


    def propagate(self):
        """ This method invokes the propagation on the current model.

        See documentation of CpoSolver.propagate() for details.

        Returns:
            Propagation result,
            object of class :class:`~docplex.cp.solution.CpoSolveResult`.
        """
        # Request propagation
        self._write_message(CMD_PROPAGATE)

        # Wait JSON result
        jsol = self._wait_json_result(EVT_PROPAGATE_RESULT)

        # Build result object
        return self._create_result_object(CpoSolveResult, jsol)


    def end(self):
        """ End solver and release all resources.
        """
        if self.process is not None:
            try:
                self._write_message(CMD_EXIT)
            except:
                pass
            time.sleep(0.300)
            try:
                self.pout.close()
            except:
                pass
            try:
                self.pin.close()
            except:
                pass
            try:
                self.process.kill()
            except:
                pass
            self.process = None
            super(CpoSolverLocal, self).end()


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
                if self.log_enabled and data:
                    self._add_log_data(data.decode('utf-8'))
            elif evt == EVT_SOLVER_ERR_STREAM:
                if data:
                    ldata = data.decode('utf-8')
                    if firsterror is None:
                        firsterror = ldata.replace('\n', '')
                    out = self.log_output if self.log_output is not None else sys.stdout
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
                raise LocalSolverException("Solver error: " + errmsg)
            else:
                self.end()
                raise LocalSolverException("Unknown event received from solver angel: " + str(evt))

        # Return
        return data


    def _wait_json_result(self, evt):
        """ Wait for a JSON result while forwarding logs if any.
        Args:
            evt: Event to wait for
        Returns:
            JSON solution string, decoded from UTF8
        """

        # Wait JSON result
        data = self._wait_event(evt)
        self.process_infos[CpoProcessInfos.RESULT_DATA_SIZE] = len(data)

        # Decode json result
        stime = time.time()
        self.last_json_result = data.decode('utf-8')
        self.process_infos[CpoProcessInfos.RESULT_DECODE_TIME] = time.time() - stime

        return self.last_json_result


    def _write_message(self, cid, data=None):
        """ Write a message to the solver process
        Args:
            cid:   Command name
            data:  Data to write, already encoded in UTF8 if required
        """
        # Build header
        cid = cid.encode('utf-8')
        tlen = len(cid)
        if data is not None:
            tlen += len(data) + 1
        if tlen > 0xffffffff:
            raise LocalSolverException("Try to send a message with length {}, greater than {}.".format(tlen, 0xffffffff))
        frame = bytearray(6)
        frame[0] = 0xCA
        frame[1] = 0xFE
        frame[2] = (tlen >> 24) & 0xFF
        frame[3] = (tlen >> 16) & 0xFF
        frame[4] = (tlen >> 8)  & 0xFF
        frame[5] = tlen         & 0xFF

        # Add data if any
        self.context.log(5, "Send message: cmd=", cid, ", tsize=", tlen)
        if data is None:
            frame = frame + cid
        else:
            frame = frame + cid + bytearray(1) + data

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
            # print("Wrong input: {}{}".format(frame, self._read_frame(150)))
            self.end()
            raise LocalSolverException("Invalid message header '{}'. Probable wrong destination or desynchronization of stream.".format(frame))

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
                raise LocalSolverException("Nothing to read from angel process.")
            else:
                raise LocalSolverException("Read only {} bytes when {} was expected.".format(len(data), nbb))
        # Return
        if IS_PYTHON_2:
            return bytearray(data)
        return data


# For ascending compatibility
CpoSolverAngel = CpoSolverLocal





