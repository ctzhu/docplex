# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.lp_printer import LPModelPrinter


class ModelPrinterFactory(object):
    __printer_ext_map = \
        {printer().get_format(): printer for printer in {LPModelPrinter}}
    default_printer_type = LPModelPrinter

    @staticmethod
    def new_printer(exchange_format, error_handler, hide_user_names=False):
        """
        returns a new printer
        :param exchange_format:
        :param error_handler:
        :param hide_user_names:
        :return:
        """
        printer_type = ModelPrinterFactory.__printer_ext_map.get(exchange_format,
                                                                 ModelPrinterFactory.default_printer_type)
        if not printer_type:
            error_handler.fatal("Unsupported output format: {0!s}", exchange_format)
        else:
            printer = printer_type()
            printer.forget_user_names = hide_user_names
            return printer
