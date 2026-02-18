from __future__ import annotations
import sys

import argparse
from pathlib import Path

import utils

class ArgParser():
    def __init__(self) -> None:
        p = argparse.ArgumentParser(prog=sys.argv[0], description="FCI grid generator")
        p.add_argument("--plot", action=argparse.BooleanOptionalAction,
                default=True, help="Plot 2D device configuration/field.")
        p.add_argument("--debug", action=argparse.BooleanOptionalAction,
                default=False, help="Enable debug logging and plotting.")
        p.add_argument("--gfile", type=str, default=str(utils.DEFAULT_GFILE),
                help=("EQDSK file name. Examples: 'TokData/DIIID/g174791.03000', 'TokData/TCV/65402_t1.eqdsk'."))

        #TODO: Allow for passing single perp resolution --n? Match x and z before ghosts, dont need x==z as in zoidberg?
        p.add_argument("--nx", type=int, default=utils.DEFAULT_NX, help="Horizontal gridpoints (x).")
        p.add_argument("--ny", type=int, default=utils.DEFAULT_NY, help="Parallel gridpoints (y).")
        p.add_argument("--nz", type=int, default=utils.DEFAULT_NZ, help="Vertical gridpoints (z).")

        #Allow for custom domain if needed. Useful for MMS testing.
        p.add_argument("--x0", type=float, default=utils.DEFAULT_START, help="Left end of domain.")
        p.add_argument("--x1", type=float, default=utils.DEFAULT_STOP,  help="Right end of domain.")
        p.add_argument("--z0", type=float, default=utils.DEFAULT_START, help="Top of domain.")
        p.add_argument("--z1", type=float, default=utils.DEFAULT_STOP,  help="Bottom of domain.")

        self.parser = p

    #Define function to test grid resolutions. Prefer powers of two.
    def _pow2_int(self, s: str) -> int:
        #Make sure its an integer.
        try:
            v = int(s)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))

        #Make sure valid integer and prefer powers of 2.
        if v <= 0:
            raise argparse.ArgumentTypeError("must be >= 1")
        if (v & (v - 1)) != 0:
            utils.logger.warn(f"Prefer powers of two for number of gridpoints. Found: {s}.")

    #Function to test gfile valid.
    def _gfile_valid(self, gfile: str) -> Path:
        gpath = Path(gfile).resolve(strict=False) #Don't require existence yet.
        if not gpath.is_file():
            raise SystemExit(f"Error: gfile not found: {gpath}")
        return gpath

    def parse(self, argv: list[str] | None = None) -> argparse.Namespace:
        """Parse argv (or sys.argv) and return args. args.gfile is a Path."""
        args = self.parser.parse_args(argv)

        #Validate parameters.
        gpath = self._gfile_valid(args.gfile)
        args.gfilename = gpath.name #Store the filename from the path.
        self._pow2_int(args.nx)
        self._pow2_int(args.ny)
        self._pow2_int(args.nz)

        return args
