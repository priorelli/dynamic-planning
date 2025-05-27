# Dynamic planning

<p align="center">
  <img src="/reference/images/env.png">
</p>

This is the project related to the papers [Deep hybrid models: infer and plan in a dynamic world](https://www.mdpi.com/1099-4300/27/6/570) and [Dynamic planning in hierarchical active inference](http://sciencedirect.com/science/article/pii/S0893608024010049). The first paper describes a (deep) hierarchical hybrid approach that can plan efficiently in constantly changing environments. The proposed method is based on three caracteristics: (i) the capacity to understand and exploit affordances for object manipulation; (ii) to learn the hierarchical interactions between the self and the environment, including other agents; (iii) to relate continuous trajectories to discrete plans. The second paper provides a review for many tasks in motor control, such as object grasping, obstacle avoidance, multi-agent interaction, and tool use.

Video simulations are found [here](https://priorelli.github.io/projects/6_dynamic_planning/).

Check [this](https://priorelli.github.io/blog/) and [this](https://priorelli.github.io/projects/) for additional guides and projects.

This study has received funding from the [MAIA project](https://www.istc.cnr.it/it/content/maia-multifunctional-adaptive-and-interactive-ai-system-acting-multiple-contexts), under the European Union's Horizon 2020 Research and Innovation programme.

## HowTo

### Start the simulation

The simulation can be launched through *main.py*, either with the option `-m` for manual control, `-i` for the active inference agent. If no option is specified, the last one will be launched. For the manual control simulation, the arm can be moved with the keys `Z`, `X`, `A`, `S`, `LEFT`, `RIGHT`, `UP` and `DOWN`.

Plots can be generated through *plot.py*, either with the option `-d` for the belief trajectories, or `-v` for generating a video of the simulation.

The tasks are organized as follows:

- *2.1_simple_agent/*: a basic 1-DoF agent that has to reach a fixed position;
- *2.2_tracking_objects/*: a 1-DoF agent that has to track a moving object;
- *2.3_object_affordances/*: a 3-DoF agent that maintains distinct body potential configurations for each object and distinct potential trajectories for each intention; this agent has to reach the red ball and then the green square;
- *3.1_intrinsic_extrinsic/*: a hierarchical 3-DoF agent that has to reach the red ball while avoiding the green square;
- *3.2_deep_hierarchies/*: a (deep) hierarchical 8-DoF agent that has to reach the red ball with a finger;
- *3.3_self_others/*: a 3-DoF agent that has to reach the elbow of another 4-DoF agent, which in turn has to reach the hand of the first agent;
- *4.1_dynamic_inference/*: a 1-DoF agent moving along a circular trajetory and has to infer which one of the two objects is following;
- *4.2_dynamic_planning/*: an 8-DoF agent that has to grasp a moving ball;
- *4.3_flexible_hierarchies/*: a 4-DoF agent that has to grasp a moving tool and reach a moving ball with the tool's extremity.

### Advanced configuration

More advanced parameters can be manually set from *config.py*. Custom log names are set with the variable `log_name`. The number of trials and steps can be set with the variables `n_trials` and `n_steps`, respectively.

The parameter `n_tau` specifies the sampling time window used for evidence accumulation, while `n_policy` controls the length of the policies.

The parameter `n_objects` specifies the number of objects, which are defined in the class `Objects` in *environment/objects.py*.

The parameter `lr_len` controls the learning rate of the length beliefs. If it is set to 0, these beliefs will not be updated.

### Agent

The script *simulation/inference.py* contains a subclass of `Window` in *environment/window.py*, which is in turn a subclass `pyglet.window.Window`. The only overriden function is `update`, which defines the instructions to run in a single cycle. Specifically, the subclass `Inference` initializes the agent and the objects; during each update, it retrieves proprioceptive and visual observations through functions defined in *environment/window.py*, calls the function `inference_step` of the agent, and finally moves the arm and the objects.

The script *brain.py* defines the structure of the (deep) hierarchical hybrid model, along with the dynamics functions. The script *ie.py* contains the `IE` class corresponding to an Intrinsic/Extrinsic module, and defines the proprioceptive, visual, and extrinsic likelihood functions. The script *unit.py* contains the `Unit` class specifiying a single hybrid unit, and the `Obs` class defining an observation modality. These modular units can be linked in a multiple input and multiple output architecture as defined by the `IE` and `Brain` classes. Sensory evidence is accumulated in every unit until the next discrete step. Finally, the discrete model is described in *discrete.py*, which performs Bayesian model comparison depending on the hybrid units, and generate the new discrete hidden causes.

Every computation is performed by automatic differentiation through `pytorch`.

Useful trajectories computed during the simulations are stored through the class `Log` in *environment/log.py*.

Note that all the variables are normalized between -1 and 1 to ensure that every contribution to the belief updates has the same magnitude.

## Required libraries

matplotlib==3.9.2

numpy==2.1.3

pyglet==2.0.18

pymunk==6.6.0

seaborn==0.13.2

torch==2.3.1
