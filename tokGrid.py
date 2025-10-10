#!/usr/bin/env python

from importlib.metadata import version
import sys
import uuid

from pathlib import Path

from boututils import datafile as bdata
from hypnotoad import __version__ #TODO: Dont depend on hypnotoad.

from data import TokamakData
from grid import StructuredPoloidalGrid
from field import MagneticField
from fci_parser import ArgParser #TODO: parser taken by std lib, pick better name?
import utils
import weights

#TODO:Add unit tests to make sure functionality is reasonable? Convert to classes with type hints, type checks, and header comments?

def write_output(gfilename, tok_grid: StructuredPoloidalGrid, tok_field: MagneticField,
            attributes, maps, metric):
    #Write output to data file.
    gridfile = gfilename + ".fci.nc"
    print("Writing to " + str(gridfile) + "...")
    with bdata.DataFile(gridfile, write=True, create=True, format="NETCDF4") as f:
        f.write_file_attribute("title", "BOUT++ FCI grid file")
        f.write_file_attribute("software_name", "fci-grid")
        f.write_file_attribute("software_version", __version__)
        grid_id = str(uuid.uuid1())
        f.write_file_attribute("id", grid_id)      #Conventional name
        f.write_file_attribute("grid_id", grid_id) #BOUT++ specific name

        f.write("nx", tok_grid.nr)
        f.write("ny", tok_grid.nphi)
        f.write("nz", tok_grid.nz)

        f.write("dx", metric["dx"])
        f.write("dy", metric["dy"])
        f.write("dz", metric["dz"])

        f.write("ixseps1", tok_grid.ixseps1)
        f.write("ixseps2", tok_grid.ixseps2)

        f.write("B", tok_grid.make_3d(tok_field.Bmag))
        f.write("pressure", tok_grid.make_3d(tok_field.pres))

        for key, value in metric.items():
            f.write(key, value)

        for key, value in attributes.items():
            f.write(key, value)

        for key, value in maps.items():
            f.write(key, value)

def main(args):
    #Set up global debug flags if necessary.
    if args.debug:
        utils.DEBUG_FLAG = True
        utils.logger.min_level = utils.logger.Level.DEBUG

    #Read eqdsk file.
    tokData = TokamakData(args.gfile)

    #Generate grid.
    tok_grid  = StructuredPoloidalGrid(tokData, args.nr, args.nphi, args.nz)
    tok_field = MagneticField(tokData, tok_grid)
    tok_grid.attach_field(tok_field) #Get around circular import...

    maps, metrics = tok_grid.generate_maps()

    #Generate perp weights from ghost points/BCs.
    tok_grid.generate_bounds(maps)
    #Generate parallel weights.
    #TODO: Not used by BOUT++ yet. Uses hermite-spline interp by default.
    par_wghts = weights.calc_par_weights(maps)

    psi = tok_field.psi
    attributes = {
        "psi": tok_grid.make_3d(psi)
    }

    write_output(args.gfilename, tok_grid, tok_field, attributes, maps, metrics)

    if (args.plot):
        tok_grid.plotConfig(psi, maps)

if __name__ == "__main__":
    args = ArgParser().parse()
    main(args)