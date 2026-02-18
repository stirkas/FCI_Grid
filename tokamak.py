from tokamak_config import TokamakConfig
from fci_parser import ArgParser
import utils

def main(args) -> None:
    #Set up global debug flags if necessary.
    if args.debug:
        utils.DEBUG_FLAG = True
        utils.logger.min_level = utils.logger.Level.DEBUG

    cfg = TokamakConfig(
        nx=args.nx, nz=args.nz, ny=args.ny,
        gfile=args.gfile, show_plot=args.plot) #TODO: Only needed if args.plot==False?

    cfg.generate()

if __name__ == "__main__":
    args = ArgParser().parse()
    main(args)