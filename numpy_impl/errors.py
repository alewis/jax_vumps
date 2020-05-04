
"""
ERROR MESSAGES
Functions here take a value 'val' and a name (the name of the variable).
They perform a test on val, and return a tuple (errflag, errstring). 
Errflag is False iff the test passed. If it failed, errflag is True,
and errstring contains an appropriate error message.
"""


def check_natural(val, name: str):
    """
    Passes when val is a natural number (an integer greater than 0).
    """
    flag = False 
    errstr = ""
    if val != round(val) or val <= 0:
        flag = True 
        errstr = name + " = " + val + " must be a natural number."
    return (flag, errstr)


