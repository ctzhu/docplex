# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore
from collections import Counter


class ExprCounter(Counter):
    """
    A subclass of Counter which does not require a dictionary to be updated
    Can be updated from an item (assumed to be a key)
    or from a key and a value
    SEE how to remember the order in which objects are added.
    """

    def update_from_item_value(self, item, value=1, _dict_get=dict.get, _dict_setitem=dict.__setitem__):
        """
        This differs from standard Counter when a dict instance is required.
        :param item: the key to be updated
        :param value: the associated value
        :return:
        """
        if value:
            if item in self:
                new_value = _dict_get(self, item, 0) + value
                if 0 == new_value:
                    del self[item]
                else:
                    _dict_setitem(self, item, new_value)
            else:
                _dict_setitem(self, item, value)

    def normalize(self, _dict_get=dict.get):
        """
        Removes all entries with zero value
        :return:
    """
        doomed_keys = [k for k in self if _dict_get(self, k) is 0]
        for dk in doomed_keys:
            del self[dk]
        return self
