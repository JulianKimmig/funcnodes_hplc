import funcnodes as fn
from funcnodes_span import NODE_SHELF as SPAN_SHELF
from .report import REPORT_SHELF

NODE_SHELF = fn.Shelf(
    nodes=[],
    name="Funcnodes Hplc",
    description="The nodes of Funcnodes Hplc package",
    subshelves=[REPORT_SHELF, SPAN_SHELF],
)
