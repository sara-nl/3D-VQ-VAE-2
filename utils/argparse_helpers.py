
def booltype(inp: str) -> bool:
    if type(inp) is str:
        if inp.lower() == 'true':
            return True
        elif inp.lower() == 'false':
            return False

    raise ValueError(f"input should be either 'True', or 'False', found {inp}")