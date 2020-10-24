import os
import csv
import time
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import Sequence
from datetime import datetime
from collections import Counter

from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# Agent SEIRU States
SUSCEPTIBLE = "susceptible"
EXPOSED = "exposed"
INFECTED = "infected"
SYMPTOMATIC = "symptomatic"
REPORTED = "reported"
ASYMPTOMATIC = "asymptomatic"
UNREPORTED = "unreported"
RECOVERED = "recovered"

AGENT_STATES = frozenset({
    SUSCEPTIBLE, EXPOSED, INFECTED, REPORTED, UNREPORTED,
    SYMPTOMATIC, ASYMPTOMATIC, RECOVERED,
})

REPORT_STATES = frozenset({
    SUSCEPTIBLE, EXPOSED, INFECTED, REPORTED, UNREPORTED, RECOVERED
})

COUNTDOWN_STATES = frozenset({
    EXPOSED, INFECTED, SYMPTOMATIC, REPORTED, ASYMPTOMATIC, UNREPORTED,
})

CONTAGIOUS_STATES = frozenset({
    INFECTED, SYMPTOMATIC, REPORTED, UNREPORTED,
})


class OperaAgent(Agent):

    # Autoincrementing counter for agent IDs
    IDSEQ = Sequence()

    def __init__(self, model, state=SUSCEPTIBLE, **kwargs):
        unique_id = OperaAgent.IDSEQ()
        super(OperaAgent, self).__init__(unique_id, model)

        # Get probabilities and durations from kwargs
        # TODO: create viral load mechanism
        # Temporary: transition probabilities instead of viral load
        self.exposure_probability = kwargs.pop("exposure_probability", 1.0)
        self.reported_probability = kwargs.pop("reported_probability", 0.5)
        self.immunity_probability = kwargs.pop("immunity_probability", 1.0)

        # Defines durations each agent remains in the state
        self.exposure_duration_dist = kwargs.pop("exposure_duration_dist", (1, 3))
        self.infected_duration_dist = kwargs.pop("infected_duration_dist", (1, 14))
        self.symptoms_duration_dist = kwargs.pop("symptoms_duration_dist", (6, 21))

        # Set the state of the agent
        self.state = state
        self.location = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        if val is  None or val not in AGENT_STATES:
            raise ValueError(f"'{val}' is not a valid agent state")
        self._state = val

        # If the state is time limited, compute how long the agent will remain in state
        if self._state in COUNTDOWN_STATES:
            if self._state == EXPOSED:
                dist = self.exposure_duration_dist
                self.state_duration = self.random.randrange(*dist)
            elif self._state == INFECTED:
                dist = self.infected_duration_dist
                self.state_duration = self.random.randrange(*dist)
            elif self._state == REPORTED or self._state == UNREPORTED:
                dist = self.symptoms_duration_dist
                self.state_duration = self.random.randrange(*dist)
            else:
                raise TypeError("unhandled countdown state '{self._state}'")

    def step(self):
        self.move()
        if self.state in COUNTDOWN_STATES:
            # NOTE: could this mean that the exposed state might last 0 timesteps if
            # the exposure selects 1 timestep duration but the agent was exposed before
            # their step function in the current timestep was called.
            self.state_duration -= 1
            if self.state_duration <= 0:
                # Transition to the next state!
                if self.state == EXPOSED:
                    self.state = INFECTED
                elif self.state == INFECTED:
                    if self.random.random() <= self.reported_probability:
                        self.state = REPORTED
                    else:
                        self.state = UNREPORTED
                elif self.state == REPORTED or self.state == UNREPORTED:
                    if self.random.random() <= self.immunity_probability:
                        self.state = RECOVERED
                    else:
                        self.state = SUSCEPTIBLE
                else:
                    raise TypeError("unhandled state transition from '{self._state}'")

        if self.state in CONTAGIOUS_STATES:
            self.infectious()


    def move(self):
        dst = self.model.world.destination(self.location)
        if dst != self.location:
            self.model.world.move_agent(self, dst)
            self.location = dst

    def expose(self):
        if self.state == SUSCEPTIBLE and self.exposure_probability > 0:
            if self.random.random() <= self.exposure_probability:
                self.state = EXPOSED

    def infectious(self):
        i = 0
        n = self.random.randrange(3)
        for agent in self.model.world.get_cell_list_contents([self.pos]):
            if agent == self or not agent.state == SUSCEPTIBLE:
                continue

            i += 1
            agent.expose()
            if i >= n:
                return


class OperaModel(Model):

    def __init__(self, S, I=1, G=None):
        self.num_agents = S+I
        self.world = GraphWorld(G or generate_world_graph())
        self.schedule = RandomActivation(self)

        # Create agents
        nodes = list(self.world.G.nodes())
        for i in range(self.num_agents):
            state = INFECTED if i < I else SUSCEPTIBLE
            agent = OperaAgent(self, state=state)
            self.schedule.add(agent)

            # Add agents to a random grid cell
            node = self.random.choice(nodes)
            self.world.place_agent(agent, node)
            agent.location = node

        # Create monitor
        self.monitor = DataCollector(
            model_reporters={"states": sum_states},
        )

    def step(self):
        self.monitor.collect(self)
        self.schedule.step()

    def results(self):
        return pd.DataFrame(self.monitor.model_vars["states"])


class GraphWorld(NetworkGrid):

    @property
    def size(self):
        return self.G.order()

    def destination(self, src):
        """
        Uses the network to select a destination from the given source node based on
        the travel probabilities assigned to outgoing edges. The destination may be the
        src node (e.g. do not travel) if the sum of the outgoing probabilities is < 1.
        """
        p = self.G.nodes[src]["prob"]
        idx = np.searchsorted(p, np.random.rand())

        # TODO: cache number of neighbors for faster lookup
        nbrs = list(self.G.neighbors(src))
        if idx == len(nbrs):
            return src
        return nbrs[idx]


def sum_states(model):
    counter = Counter(dict(zip(REPORT_STATES, [0]*len(REPORT_STATES))))
    for agent in model.schedule.agents:
        counter[agent.state] += 1
    return counter


def generate_world_graph(n=6, p=1, q=4, r=2):
    """
    Uses nx.navigable_small_world_graph to generate a directed grid with additional
    long range connections that are chosen randomly. For each node, each out edge is
    assigned a probability such that the sum of all probabilities <= 1.0 (where the
    remainder is the probability of staying in the node). Nodes are also assigned unique
    random place names and given data attributes for use in the Graph.
    """
    G = nx.navigable_small_world_graph(n, p=p, q=q, r=r, dim=2)
    for node in G.nodes():
        # Assign probabilities to all edges such that probabilities are <= 1
        # TODO: ensure there isn't an edge to self
        dist = np.random.dirichlet(np.ones(G.out_degree(node) + 1), size=1)[0]
        for i, edge in enumerate(G.out_edges(node)):
            G.edges[edge]["weight"] = dist[i]
            G.edges[edge]["distance"] = np.linalg.norm(
                np.asarray(edge[0]) - np.asarray(edge[1])
            )

        # Assign the travel distribution to the node for selection
        G.nodes[node]["prob"] = np.cumsum(dist)

    G.name = "Navigable Small World"
    return G


##########################################################################
## Commands
##########################################################################

def run(args):
    world = generate_world_graph(n=args.world)
    model = OperaModel(S=args.agents-1, I=1, G=world)
    for _ in tqdm(range(args.timesteps)):
        model.step()

    results = model.results()

    if not args.no_show:
        _, ax = plt.subplots(figsize=(9,6))
        results.plot(ax=ax)
        ax.set_title(f"{model.num_agents} agents in an order {model.world.size} world")
        ax.set_xlabel("timestep")
        ax.set_ylabel("number of agents")
        plt.show()


def bench(args):
    name = f"mesasimbench{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    outpath = os.path.join(args.dir, name)
    with open(outpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["nodes", "agents", "runtime", "version", "timesteps"])

        niters = (args.maxa - args.mina + 1) * (args.maxw - args.minw + 1)
        with tqdm(total=niters) as pbar:
            for w in range(args.minw, args.maxw + 1):
                for a in range(args.mina, args.maxa + 1):
                    # TODO: add parallelism
                    nagents = a * (w ** 2)
                    model = OperaModel(S=nagents-1, I=1, G=generate_world_graph(n=w))

                    start = time.time()
                    for _ in range(args.timesteps):
                        model.step()

                    writer.writerow(
                        [
                            model.world.size,
                            model.num_agents,
                            time.time() - start,
                            "mesa",
                            args.timesteps,
                        ]
                    )
                    f.flush()
                    pbar.update(1)



##########################################################################
## CLI Parsing and Main
##########################################################################

DATA_HOME_ENV = "ABSIM_DATA_HOME"
PROG = "opera"
DESCRIPTION = "reimplementation of SEIRU using MESA"
EPILOG = "please post issues or bugs to GitLab"
VERSION = "v0.1-alpha"


def make_wide(formatter, width=120, max_help_position=42):
    """
    Increase space between arguments and help text, if possible.
    See: https://stackoverflow.com/questions/5462873/
    """
    try:
        # ensure formatter can be used
        kwargs = {"width": width, "max_help_position": max_help_position}
        formatter(None, **kwargs)

        # return function to create formatter
        return lambda prog: formatter(prog, **kwargs)
    except TypeError:
        # return originally sized formatter
        return formatter


if __name__ == "__main__":
    cmds = {
        "run": {
            "func": run,
            "description": "run the zombies simulation",
            "args": {
                ("-w", "--world"): {
                    "type": int,
                    "metavar": "N",
                    "default": 6,
                    "help": "size of world grid, e.g. number of nodes is N**2",
                },
                ("-a", "--agents"): {
                    "type": int,
                    "metavar": "N",
                    "default": 2500,
                    "help": "number of agents in the system, random by default",
                },
                ("-t", "--timesteps"): {
                    "type": int,
                    "metavar": "T",
                    "default": 180,
                    "help": "number of timesteps to run the simulation for",
                },
                ("-S", "--no-show"): {
                    "action": "store_true",
                    "help": "do not show the results figure on completion",
                }
            },
        },
        "bench": {
            "func": bench,
            "description": "benchmark the simulation by size of world",
            "args": {
                ("-w", "--minw"): {
                    "type": int,
                    "metavar": "N",
                    "default": 2,
                    "help": "minimum world size to start benchmark from",
                },
                ("-W", "--maxw"): {
                    "type": int,
                    "metavar": "N",
                    "default": 9,
                    "help": "maximum world size to benchmark to",
                },
                ("-a", "--mina"): {
                    "type": int,
                    "metavar": "N",
                    "default": 50,
                    "help": "minimum number of agents per node in the world",
                },
                ("-A", "--maxa"): {
                    "type": int,
                    "metavar": "N",
                    "default": 100,
                    "help": "maximum number of agents per node",
                },
                ("-t", "--timesteps"): {
                    "type": int,
                    "metavar": "T",
                    "default": 365,
                    "help": "number of timesteps to run each simulation",
                },
                ("-d", "--dir"): {
                    "type": str,
                    "metavar": "PATH",
                    "default": os.getenv(DATA_HOME_ENV, os.getcwd()),
                    "help": f"specify results folder [${DATA_HOME_ENV} or $CWD]",
                },
            },
        },
    }

    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION, epilog=EPILOG)
    parser.add_argument("-v", "--version", action="version", version=VERSION)
    subparsers = parser.add_subparsers(title="commands")

    for cmd, cmda in cmds.items():
        subp = subparsers.add_parser(
            name=cmd,
            description=cmda["description"],
            formatter_class=make_wide(argparse.ArgumentDefaultsHelpFormatter),
        )
        subp.set_defaults(func=cmda["func"])

        for pargs, kwargs in cmda["args"].items():
            if isinstance(pargs, str):
                pargs = (pargs,)
            subp.add_argument(*pargs, **kwargs)

    args = parser.parse_args()
    if "func" in args:
        try:
            args.func(args)
        except Exception as e:
            parser.error(e)
    else:
        parser.print_help()
