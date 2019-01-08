# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore
from collections import Counter, OrderedDict
from six import iteritems


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


class FastOrderedDict(dict):
    """ A subclass of dict that keeps key ordering.

    """

    def __init__(self, init_val=None):
        """
        Create a new ordered dictionary. Cannot init from a normal dict,
        nor from kwargs, since items order is undefined in those cases.

        """
        # noinspection PyTypeChecker
        dict.__init__(self)
        self._key_seq = []
        if init_val is None:
            pass
        elif isinstance(init_val, FastOrderedDict):
            self._key_seq = init_val.keys()
            dict.update(self, init_val)
        elif isinstance(init_val, dict):
            len_initial = len(init_val)
            # ok for dicts of size 1
            if len_initial > 1:
                # we lose compatibility with other ordered dict types this way
                raise TypeError('undefined order, cannot get items from dict')
            elif len_initial == 0:
                pass
            else:
                # len is 1
                self._update_from_key_values([(k, init_val[k]) for k in init_val])
        else:
            # init_val is assumed to be a list of (key, value) tuples
            key_list = [kv[0] for kv in init_val]
            if len(set(key_list)) == len(key_list):
                for k, v in init_val:
                    dict.__setitem__(self, k, v)
                self._key_seq = key_list
            else:
                self._update_from_key_values(init_val)

    def __delitem__(self, key):
        self.remove(key)

    def remove(self, key):
        """ Removes a key from the dict; updates key sequence
        :param key:
        """
        try:
            dict.__delitem__(self, key)
            # find the index without calling == otherwise create constraints...
            key_index = self._index_is(key)
            if key_index >= 0:
                del self._key_seq[key_index]
        except KeyError:
            pass

    def equals(self, other):
        if len(self) != len(other):
            return False
        self_iteritems = self.iteritems()
        other_iteritems = other.iteritems()
        try:
            while True:
                self_k, self_v = next(self_iteritems)
                other_k, other_v = next(other_iteritems)
                if self_k != other_k:
                    return False
                elif self_v != other_v:
                    return False
                else:
                    continue
        except StopIteration:
            pass
        return True

    def __eq__(self, other):
        if isinstance(other, FastOrderedDict):
            return self.equals(other)
        else:
            return False

    def __repr__(self):
        return '%s([%s])' % (self.__class__.__name__, ', '.join(
            ['(%r, %r)' % (key, self[key]) for key in self._key_seq]))

    def __setitem__(self, key, val, dict_setitem=dict.__setitem__):
        if key not in self:
            self._key_seq.append(key)
        dict_setitem(self, key, val)

    __str__ = __repr__

    def __deepcopy__(self, memo):
        """
        To allow deepcopy to work with OrderedDict.
        """
        from copy import deepcopy

        return self.__class__(deepcopy(self.items(), memo))

    def copy(self):
        """ returns a copy of the ordered dict.
        """
        return FastOrderedDict(self)

    def items(self):
        """ Returns a lits of (key, value) tuples in sequence order.

        """
        return [(k, self[k]) for k in self._key_seq]

    def keys(self):
        """ Returns a copy of the ordered keys.
        """
        return self._key_seq[:]

    # noinspection PyMethodOverriding
    def values(self):
        """ Return a list of values in key sequence order.
        """
        return [self[key] for key in self._key_seq]

    def iteritems(self):
        """ Returns an iterator on key, value pairs, properly ordered
        """

        def generate_items(od):
            keyseq = od._key_seq
            nb_keys = len(keyseq)
            k = 0
            while k < nb_keys:
                key = keyseq[k]
                k += 1
                yield (key, self[key])

        return generate_items(self)

    def iterkeys(self):
        """ Returns an iterator on the ordered keys.
        """
        return iter(self._key_seq)

    __iter__ = iterkeys

    def itervalues(self):
        """ Returns an iterator over values, properly ordered
        """
        def generate_values(od):
            iter_keys = od.iterkeys()
            while True:
                yield od[iter_keys.next()]

        return generate_values(self)

    def clear(self):
        """ Clears the dict, and its key sequence
        """
        dict.clear(self)
        self._key_seq = []

    def update(self, from_arg):
        """
        Update from another OrderedDict or sequence of (key, value) pairs

        """
        if isinstance(from_arg, (FastOrderedDict, OrderedDict)):
            for key, val in from_arg.items():
                self[key] = val
        elif isinstance(from_arg, dict):
            # we lose compatibility with other ordered dict types this way
            raise TypeError('undefined order, cannot get items from dict')
        else:
            self._update_from_key_values(from_arg)

    def update_from_item_value(self, item, value=1,
                               _dict_get=dict.get, dict_set=dict.__setitem__,
                               fastdict_set=__setitem__):
        """
        This differs from standard Counter when a dict instance is required.
        :param item: the key to be updated
        :param value: the associated value
        :return:
        """
        if value:
            old_value = _dict_get(self, item, 0)
            if old_value:
                new_value = old_value + value
                if 0 != new_value:
                    dict_set(self, item, new_value)
                else:
                    del self[item]
            else:
                # we are sure item is not already present, it must be added to the sequence...
                fastdict_set(self, item, value)


    def _update_from_key_values(self, kv_seq):
        for key, val in kv_seq:
            self[key] = val

    def _index_is(self, key):
        # returns the index of key in the seqeunce without calling ==, but "is"
        # returns -1 if not found
        idx = 0
        hash_key = hash(key)
        for k in self._key_seq:
            if k is key or hash(k) == hash_key:
                return idx
            else:
                idx += 1
        else:
            return -1

    def reverse(self):
        """
        Reverse the order of the keys.

        """
        self._key_seq.reverse()

    def sort(self, *args, **kwargs):
        """ Applies a sort to the key sequence.

        The underlying dict is not modified.

        """
        self._key_seq.sort(*args, **kwargs)


    def __reduce__(self):
        'Return state information for pickling'
        items = [[k, self[k]] for k in self]
        inst_dict = vars(self).copy()
        for k in vars(OrderedDict()):
            inst_dict.pop(k, None)
        if inst_dict:
            return (self.__class__, (items,), inst_dict)
        return self.__class__, (items,)


class FastOrderedCounter(FastOrderedDict):
    """
    A subclass of Counter which does not require a dictionary to be updated
    Can be updated from an item (assumed to be a key)
    or from a key and a value
    """

    def __init__(self, e=None):
        dict.__init__(self)
        self._key_seq = []
        if e is None:
            pass

        elif isinstance(e, FastOrderedDict):
            self._key_seq = e.keys()
            dict.update(e)

        elif isinstance(e, list):
            try:
                for item, value in e:
                    self.update_from_item_value(item, value)
            except ValueError:
                for item in e:
                    self.update_from_item(item)
        else:
            raise TypeError

    def update_from_item(self, item, _dict_get=FastOrderedDict.get):
        """
        Adds one item occurence
        :param item:
        :return:
        """
        self[item] = _dict_get(self, item, 0) + 1

    def update_from_sequence(self, items):
        """  Updates with a sequence.

        :param items:
        :return:
        """
        for item in items:
            self.update_from_item(item)

    def update_from_item_value(self, item, value=1, _dict_get=dict.get):
        """
        This differs from standard Counter when a dict instance is required.
        :param item: the key to be updated
        :param value: the associated value
        :return:
        """
        if value:
            new_value = _dict_get(self, item, 0) + value
            if 0 != new_value:
                FastOrderedDict.__setitem__(self, item, new_value)
            else:
                del self[item]

    def update_from_scaled_dict(self, other_dict, factor, _dict_get=dict.get):
        """
        Updates counter from a dict instance, but with an inflation factor.
        Does nothing if factor is 0
        """
        if 0 == factor:
            # nothin to do
            pass
        elif 1 == factor:
            # standard update
            self.update(other_dict)
        else:
            # update by scaled value
            for item, value in iteritems(other_dict):
                if value:
                    self[item] = _dict_get(self, item, 0) + value * factor


    def update(self, iterable=None):
        if iterable is not None:
            if isinstance(iterable, (FastOrderedDict, OrderedDict)):
                # accept only ordered types/
                if self:
                    for elem, count in iterable.iteritems():
                        self.update_from_item_value(elem, count)
                else:
                    # fast path when counter is empty
                    dict.update(self, iterable)
                    self._key_seq = iterable.keys()
            elif isinstance(iterable, dict):
                raise TypeError("only ordered dict types allowed.")
            else:
                # a plain collection
                self_get = self.get
                for elem in iterable:
                    self[elem] = self_get(elem, 0) + 1

    def normalize(self, _dict_get=dict.get):
        """ Removes all entries with zero value.

        """
        doomed_keys = [k for k in self if _dict_get(self, k) is 0]
        for dk in doomed_keys:
            del self[dk]
        return self
