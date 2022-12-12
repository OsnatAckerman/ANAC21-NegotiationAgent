from collections import defaultdict

import matplotlib.pyplot as plt
from scml.scml2020 import SCML2020Agent, SCML2021World
from scml.scml2020.agents import RandomAgent, MarketAwareDecentralizingAgent, DecentralizingAgent
from scml.scml2020.components.production import DemandDrivenProductionStrategy, ProductionStrategy
from scml.scml2020.components.trading import PredictionBasedTradingStrategy
from scml.scml2020.components.prediction import MarketAwareTradePredictionStrategy
from negmas import LinearUtilityFunction
from negmas import SAOMetaNegotiatorController
from negmas import Contract
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from scml.scml2020 import TIME, QUANTITY, UNIT_PRICE
# from myagent.steady_mgr import SteadyMgr
from scml.scml2020.common import is_system_agent

__all__ = ["MyPaibiu"]
# class YetAnotherNegotiationManager:
#     """My new negotiation strategy
#
#     Args:
#         price_weight: The relative importance of price in the utility calculation.
#         time_range: The time-range for each controller as a fraction of the number of simulation steps
#     """
#



class MyPaibiuAgent(MarketAwareTradePredictionStrategy, PredictionBasedTradingStrategy,
               DemandDrivenProductionStrategy, SCML2020Agent):
    def __init__(self, *args, price_weight=0.7, time_horizon=0.1, utility_threshold=0.9, time_threshold=0.9, **kwargs,):
        super().__init__(*args, **kwargs)
        self.index: List[int] = None
        self.time_horizon = time_horizon
        self._price_weight = price_weight
        self._time_threshold = time_threshold
        self._utility_threshold = utility_threshold
        self._current_end = -1
        self._current_start = -1

    def step(self):
        super().step()
        # find the range of steps about which we plan to negotiate
        step = self.awi.current_step
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if step == 0:
            self.output_price = (
                self.awi.catalog_prices[self.awi.my_input_product] + production_cost
            ) * np.ones(self.awi.n_steps, dtype=int)
            self.input_cost = (self.awi.catalog_prices[self.awi.my_output_product] - production_cost
                              ) * np.ones(self.awi.n_steps, dtype=int)
        else:
            self.output_price[step:] = (
                max(self.awi.catalog_prices[self.awi.my_input_product] + production_cost, self.awi.trading_prices[self.awi.my_output_product]))
            self.input_cost[step:] = min(self.awi.catalog_prices[self.awi.my_output_product] - production_cost, self.awi.trading_prices[self.awi.my_output_product] - production_cost)
        self._current_start = step + 1
        self._current_end = min(
            self.awi.n_steps - 1,
            self._current_start + max(1, int(self.time_horizon * self.awi.n_steps)),
        )
        if self._current_start >= self._current_end:
            return

        for seller, needed, secured, product in [
            (False, self.inputs_needed, self.inputs_secured, self.awi.my_input_product),
            (True, self.outputs_needed, self.outputs_secured, self.awi.my_output_product),
        ]:
            # find the maximum amount needed at any time-step in the given range
            needs = np.max(
                needed[self._current_start: self._current_end]
                - secured[self._current_start: self._current_end]
            )
            if needs < 1:
                continue
            # set a range of prices
            if seller:
                # for selling set a price that is at least the catalog price
                min_price = max(self.awi.catalog_prices[product], self.awi.trading_prices[product])
                price_range = (min_price, 2 * min_price)
                controller = SAOMetaNegotiatorController(ufun=LinearUtilityFunction({
                    TIME: 0.0, QUANTITY: (1 - self._price_weight), UNIT_PRICE: self._price_weight
                }))
            else:
                # for buying sell a price that is at most the catalog price
                production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
                price_range = (0,  min(self.awi.catalog_prices[product] - production_cost, self.awi.trading_prices[product] - production_cost))
                controller = SAOMetaNegotiatorController(ufun=LinearUtilityFunction({
                    TIME: 0.0, QUANTITY: (1 - self._price_weight), UNIT_PRICE: -self._price_weight
                }))

            self.awi.request_negotiations(
                not seller,
                product,
                (1, needs),
                price_range,
                time=(self._current_start, self._current_end),
                controller=controller,
            )




    def sign_all_contracts(self, contracts: List[Contract]) -> List[Optional[str]]:
        # sort contracts by time and then put system contracts first within each time-step
        signatures = [None] * len(contracts)
        contracts = sorted(
            zip(contracts, range(len(contracts))),
            key=lambda x: (
                x[0].agreement["unit_price"],
                x[0].agreement["time"],
                0
                if is_system_agent(x[0].annotation["seller"])
                or is_system_agent(x[0].annotation["buyer"])
                else 1,
                x[0].agreement["unit_price"],
            ),
        )

        sold, bought = 0, 0  # count the number of sold/bought products during the loop
        s = self.awi.current_step

        for contract, indx in contracts:
            is_seller = contract.annotation["seller"] == self.id
            q, u, t = (
                contract.agreement["quantity"],
                contract.agreement["unit_price"],
                contract.agreement["time"],
            )
            if t < s and len(contract.issues) == 3:
                continue

            if is_seller:
                # Sign the first contract when the final process is assigned
                if s == 0 and self.awi.my_output_product == (self.awi.n_products - 1):
                    signatures[indx] = self.id
                # I don't sign contracts for less than the selling price
                if u < self.output_price[t]:
                    continue

                est = 0  # Estimated number of products
                # Calculate the maximum production possible before delivery date
                for i in range(1, t - s + 1):
                    est += min(self.inputs_secured[t - i], i * self.awi.n_lines)
                est = min(est, (t - s) * self.awi.n_lines)

                available = (
                    est
                    + self.internal_state["_output_inventory"]
                    - (self.outputs_secured[s:]).sum()
                )  # Add stock and sub contracted
                # Only sign contracts that ensure production is on time.
                if available - sold > q:
                    signatures[indx] = self.id
                    sold += q

            else:
                # I don't make contracts to buy at the end of the game.
                if t > self.awi.n_steps * 3 // 4:
                    continue

                # I don't sign contracts over the buying price
                if u > self.input_cost[t]:
                    continue

                needed = self.inputs_needed[
                    self.awi.n_steps - 1
                ]  # Maximum number of products that can be produced
                if needed - bought > q:
                    signatures[indx] = self.id
                    bought += q

        return signatures

    def on_agent_bankrupt(
        self,
        agent: str,
        contracts: List[Contract],
        quantities: List[int],
        compensation_money: int,
    ) -> None:
        super().on_agent_bankrupt(agent, contracts, quantities, compensation_money)
        for contract, new_quantity in zip(contracts, quantities):
            q = contract.agreement["quantity"]
            if new_quantity == q:
                continue
            t = contract.agreement["time"]
            missing = q - new_quantity
            if t < self.awi.current_step:
                continue
            if contract.annotation["seller"] == self.id:
                self.outputs_secured[t] -= missing
                if t > 0:
                    self.outputs_needed[t - 1 :] += missing
            else:
                self.inputs_secured[t] -= missing
                self.inputs_needed[t:] += missing


    def respond_to_negotiation_request(
            self,
            initiator: str,
            issues: List["Issue"],
            annotation: Dict[str, Any],
            mechanism: "AgentMechanismInterface",
    ) -> Optional["Negotiator"]:
        # refuse to negotiate if the time-range does not intersect
        # the current range
        if not (
                issues[TIME].min_value < self._current_end
                or issues[TIME].max_value > self._current_start
        ):
            return None
        if self.id == annotation["seller"]:
            controller = SAOMetaNegotiatorController(ufun=LinearUtilityFunction({
                TIME: 0.0, QUANTITY: (1 - self._price_weight), UNIT_PRICE: self._price_weight
            }))
        else:
            if self.awi.current_step == 0:
                try:
                    needs = np.max(
                        self.inputs_needed[0: self.awi.n_steps - 1]
                        - self.inputs_secured[0: self.awi.n_steps - 1]
                        , initial=0)
                except:
                    return None
            else:
                try:
                    needs = np.max(
                        self.inputs_needed[self._current_start: self._current_end]
                        - self.inputs_secured[self._current_start: self._current_end]
                        , initial=0)
                except:
                    return None
            if needs < 1:
                return None
            controller = SAOMetaNegotiatorController(ufun=LinearUtilityFunction({
                TIME: 0.0, QUANTITY: (1 - self._price_weight), UNIT_PRICE: -self._price_weight
            }))
        return controller.create_negotiator()
    """My agent"""

    # def target_quantity(self, step: int, sell: bool) -> int:
    #     """A fixed target quantity of half my production capacity"""
    #     return self.awi.n_lines // 2
    #
    # def acceptable_unit_price(self, step: int, sell: bool) -> int:
    #     """The catalog price seems OK"""
    #     return self.awi.catalog_prices[self.awi.my_output_product] if sell else self.awi.catalog_prices[
    #         self.awi.my_input_product]
    #
    # def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
    #     """A utility function that penalizes high cost and late delivery for buying and and awards them for selling"""
    #     if is_seller:
    #         return LinearUtilityFunction((0, 0.25, 1))
    #     return LinearUtilityFunction((0, -0.5, -0.8))


def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()
#
#
# world = SCML2021World(
#     **SCML2021World.generate([RandomAgent, MarketAwareDecentralizingAgent, DecentralizingAgent, MyPaibiuAgent, SteadyMgr], n_steps=10),
#     construct_graphs=True,
# )
# world.run()
# # # world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
# # # plt.show()
# show_agent_scores(world)
# #
# # world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
# # plt.show()
#
# def run(
#         competition="std",
#         reveal_names=True,
#         n_steps=10,
#         n_configs=2,
# ):
#     """
#     **Not needed for submission.** You can use this function to test your agent.
#
#     Args:
#         competition: The competition type to run (possibilities are std,
#                      collusion).
#         n_steps:     The number of simulation steps.
#         n_configs:   Number of different world configurations to try.
#                      Different world configurations will correspond to
#                      different number of factories, profiles
#                      , production graphs etc
#
#     Returns:
#         None
#
#     Remarks:
#
#         - This function will take several minutes to run.
#         - To speed it up, use a smaller `n_step` value
#
#     """
#     competitors = [
#         RandomAgent, MarketAwareDecentralizingAgent, DecentralizingAgent, MyPaibiu
#     ]
#     start = time.perf_counter()
#     if competition == "std":
#         results = anac2021_std(
#             competitors=competitors,
#             verbose=True,
#             n_steps=n_steps,
#             n_configs=n_configs,
#         )
#     elif competition == "collusion":
#         results = anac2021_collusion(
#             competitors=competitors,
#             verbose=True,
#             n_steps=n_steps,
#             n_configs=n_configs,
#         )
#     elif competition == "oneshot":
#         # Standard agents can run in the OneShot environment but cannot win
#         # the OneShot track!!
#         from scml.oneshot.agents import GreedyOneShotAgent, RandomOneShotAgent
#
#         competitors = [
#             MyPaibiuAgent,
#             RandomOneShotAgent,
#             GreedyOneShotAgent,
#         ]
#         results = anac2021_oneshot(
#             competitors=competitors,
#             verbose=True,
#             n_steps=n_steps,
#             n_configs=n_configs,
#         )
#     else:
#         raise ValueError(f"Unknown competition type {competition}")
#     # just make agent types shorter in the results
#     results.total_scores.agent_type = results.total_scores.agent_type.str.split(
#         "."
#     ).str[-1]
#     # show results
#     print(tabulate(results.total_scores, headers="keys", tablefmt="psql"))
#     print(f"Finished in {humanize_time(time.perf_counter() - start)}")
#
#
# if __name__ == "__main__":
#     # will run a short tournament against two built-in agents. Default is "std"
#     # You can change this from the command line by running something like:
#     # >> python3 paibiu.py collusion
#     import sys
#
#     run(sys.argv[1] if len(sys.argv) > 1 else "std")
