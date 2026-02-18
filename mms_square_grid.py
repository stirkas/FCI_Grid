from bndry_config import RectBoundaryConfig
from fci_parser import ArgParser
from uniform_linear_config import UniformLinearConfig
import utils

def main(args):
    #Set up global debug flags if necessary.
    if args.debug:
        utils.DEBUG_FLAG = True
        utils.logger.min_level = utils.logger.Level.DEBUG

    #TODO: Still need to test code on gridpoints directly.
    cfg = UniformLinearConfig(
        nx=args.nx, nz=args.nz, ny=1,
        xmin=args.x0, xmax=args.x1, zmin=args.z0, zmax=args.z1,
        bdy_cfg = RectBoundaryConfig(mode='centers'),
        filename='square_grid')

    cfg.generate()

if __name__ == "__main__":
    args = ArgParser().parse()
    main(args)