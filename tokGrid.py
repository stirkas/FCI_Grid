#!/usr/bin/env python

import sys
import uuid

from boututils import datafile as bdata
from hypnotoad import __version__

from data import TokamakData
from grid import StructuredPoloidalGrid
from field import MagneticField
import utils

#TODO:Add unit tests to make sure functionality is reasonable? Convert to classes with type hints, type checks, and header comments?

def write_output(gfilename, tok_grid: StructuredPoloidalGrid, tok_field: MagneticField,
            attributes, maps, metric):
    #Write output to data file.
    gridfile = gfilename + ".fci.nc"
    print("Writing to " + str(gridfile) + "...")
    with bdata.DataFile(gridfile, write=True, create=True, format="NETCDF4") as f:
        f.write_file_attribute("title", "BOUT++ grid file")
        #f.write_file_attribute("software_name", "zoidberg")
        #f.write_file_attribute("software_version", __version__)
        #grid_id = str(uuid.uuid1())
        #f.write_file_attribute("id", grid_id)      #Conventional name
        #f.write_file_attribute("grid_id", grid_id) #BOUT++ specific name

        f.write("nx", tok_grid.nr)
        f.write("ny", tok_grid.nphi)
        f.write("nz", tok_grid.nz)

        f.write("dx", metric["dx"])
        f.write("dy", metric["dy"])
        f.write("dz", metric["dz"])

        f.write("ixseps1", tok_grid.ixseps1)
        f.write("ixseps2", tok_grid.ixseps2)

        for key, value in metric.items():
            f.write(key, value)
            dump_array_for_diff(f"{key}_" + "new", value)

        f.write("B", tok_grid.make_3d(tok_field.Bmag))
        dump_array_for_diff("Bmag3D_" + "new", tok_grid.make_3d(tok_field.Bmag))

        f.write("pressure", tok_grid.make_3d(tok_field.pres))

        for key, value in attributes.items():
            f.write(key, value)

        for key, value in maps.items():
            f.write(key, value)
            dump_array_for_diff(f"{key}_" + "new", value)

def main(args):
    #Read eqdsk file.
    gfile_dir = "/home/tirkas1/Workspace/TokData"
    devices = ["DIIID", "TCV"]
    gfiles  = [["g162940.02944_670", "g163241.03500", #Old QL run, old DIIID run.
                #Ben's test cases for varying Ip and B0 directions.
               "g172208.03000", "g174791.03000", "g176413.03000", "g176312.03000"],
               ["65402_t1.eqdsk"]] #TCV cases.
    dvc_num = 0
    gfl_num = 0
    gfilepath = gfile_dir + "/" + devices[dvc_num] + "/" + gfiles[dvc_num][gfl_num]
    tokData = TokamakData(gfilepath)

    #Generate grid.
    nr, nphi, nz  = 16,16,16
    tok_grid  = StructuredPoloidalGrid(tokData, nr=nr, nphi=nphi, nz=nz)
    tok_field = MagneticField(tokData, tok_grid)
    tok_grid.attach_field(tok_field) #Get around circular import...

    #Generate ghost point mask and BC information.
    #TODO: Add parallel BCs as well based on traced points from above.
    print("Generating ghost cells and boundary conditions...")
    ghosts = tok_grid.wall.handle_bounds(tok_grid.RR, tok_grid.ZZ, show=False)

    #Generate metric and maps and so on to write out for BSTING.
    print("Generating metric and map data for output file...")
    psi = tokData.psi_func(tok_grid.R, tok_grid.Z)
    attributes = {
        "psi": tok_grid.make_3d(psi)
    }

    maps, metrics = tok_grid.generate_maps()

    write_output(gfiles[dvc_num][gfl_num], tok_grid, tok_field, attributes, maps, metrics)

    plotting = False
    if (plotting):
        tok_grid.plotConfig(psi, maps, checkPts=False)

if __name__ == "__main__":
    utils.logger.min_level = utils.logger.Level.DEBUG
    main(sys.argv[1:])