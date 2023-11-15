__all__ = ["zero_rank_print"]


def zero_rank_print(s):
    if not isinstance(s, str):
        s = repr(s)
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0):
        print("### " + s)
