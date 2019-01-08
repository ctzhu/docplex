# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2016
# --------------------------------------------------------------------------


# gendoc: ignore

import six
import warnings

from docplex.mp.context import check_credentials


def get_key_in_kwargs(__context, kwargs_dict):
    """Returns the overloaded value of api_key in the specified dict.

    If a 'key'  is found, it is returned. If 'key' is not found, this
    looks up 'api_key' (compatibility mode with versions < 1.0)

    """
    key = kwargs_dict.get('key')
    if not key:
        key = kwargs_dict.get('api_key')
    if key:
        try:
            ignored_keys = __context.solver.docloud.ignored_keys
            # if string, allow comma separated form
            if isinstance(ignored_keys, six.string_types):
                values = ignored_keys.split(",")
                if key in values:
                    return None
            elif key in ignored_keys:
                return None
        except AttributeError:
            # no ignored_keys, just pass
            pass
    return key


def get_url_in_kwargs(__context, kwargs_dict):
    """Returns the overloaded value of url in the specified dict.
    """
    url = kwargs_dict.get('url')
    if url:
        try:
            ignored_urls = __context.solver.docloud.ignored_urls
            # if string, allow comma separated form
            if isinstance(ignored_urls, six.string_types):
                values = ignored_urls.split(",")
                if url in values:
                    return None
            elif url in ignored_urls:
                return None
        except AttributeError:
            # no ignored_urls, just pass
            pass
    return url


def context_must_use_docloud(__context, **kwargs):
    # NOTE: the argument CANNOT be named 'context' here as kwargs may well contain a 'context' key

    # returns True if context + kwargs require an execution on cloud
    # this happens in the following cases:
    # (i)  kwargs contains a "docloud_context" key (compat??)
    # (ii) both an explicit url and api_key appear in kwargs
    # (iv) the context's "solver.agent" is "docloud"
    # (v)  kwargs override agent to be "docloud"
    docloud_agent_name = "docloud"  # this might change
    have_docloud_context = kwargs.get('docloud_context') is not None
    have_api_key = get_key_in_kwargs(__context, kwargs)
    have_url = get_url_in_kwargs(__context, kwargs)
    context_agent_is_docloud = __context.solver.get('agent') == docloud_agent_name
    kwargs_agent_is_docloud = kwargs.get('agent') == docloud_agent_name
    return have_docloud_context \
           or (have_api_key and have_url) \
           or context_agent_is_docloud \
           or kwargs_agent_is_docloud


def context_has_docloud_credentials(context, do_warn=True):
    have_credentials = False
    if context.solver.docloud:
        have_credentials, error_message = check_credentials(context.solver.docloud)
        if error_message is not None and do_warn:
            warnings.warn(error_message, stacklevel=2)
    return have_credentials
