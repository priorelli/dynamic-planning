import utils
from simulation.manual_control import ManualControl
from simulation.inference import Inference


def main():
    # Parse arguments
    options = utils.get_sim_options()

    # Choose simulation
    if options.manual_control:
        sim = ManualControl()

    else:
        sim = Inference()

    # Run simulation
    sim.run()


if __name__ == '__main__':
    main()
