# Scalable deep reinforcement learning in the non-stationary capacitated lot sizing problem

**Authors:** Lotte Van Hezewijk, Nico Dellaert, Willem Van Jaarsveld

## Abstract

Capacitated lot sizing problems in situations with stationary and non-stationary demand (SCLSP) are very common in practice. Solving problems with a large number of items using Deep Reinforcement Learning (DRL) is challenging due to the large action space. This paper proposes a new Markov Decision Process (MDP) formulation to solve this problem, by decomposing the production quantity decisions in a period into sub-decisions, which reduces the action space dramatically. We demonstrate that applying Deep Controlled Learning (DCL) yields policies that outperform the benchmark heuristic as well as a prior DRL implementation. By using the decomposed MDP formulation and DCL method outlined in this paper, we can solve larger problems compared to the previous DRL implementation. Moreover, we adopt a non-stationary demand model for training the policy, which enables us to readily apply the trained policy in dynamic environments when demand changes.


## 1. Introduction

Many industries are facing challenges such as resource scarcity and uncertainties in customer demands. The limited availability of production capacity hinders producing large inventories to provide a buffer for the demand uncertainty. Making the right production and replenishment decisions in this context is crucial for customer satisfaction as well as the company's financial performance. While a significant amount of demand exhibits non-stationarity (Graves, 1999; Tunc et al., 2011; Strijbosch et al., 2011; Amiri et al., 2023) , the research question of how to determine the optimal lot sizes in a capacitated system with non-stationary stochastic demand (SCLSP) is yet to be answered. Due to the complex nature of the problem, optimal decisions are unattainable, and the focus is on developing approaches to compute near-optimal policies.

In recent years, Deep Reinforcement Learning (DRL) approaches have been investigated as ways of solving sequential decision-making problems that have complicated dynamics and are difficult to solve with traditional methods. While they definitely appear to be promising, there are still several challenges that arise when trying to apply such methods in practical operations management problems like lot sizing.

The first key challenge is the data that is used to learn a good policy. DRL algorithms require a lot of samples of the states, actions and rewards to find good policies. It uses these samples to estimate the 'value' of taking certain decisions. It is often impossible to use real-life data to train the model. Consider for example the case of planning production on a machine with limited capacity. Data needed to estimate the value of actions has to come from actual costs, demand observed, inventory levels, etc. Likely, this data is only available on a daily granularity. A company would need to have a very long history to provide enough data, and even then it is questionable whether all this historical data still represents the current reality for which we aim to find a good policy.

A common way to overcome this challenge is to simulate an environment to mimic reality. In this simulation environment, we draw demand from some assumed stochastic distribution, and define some parameters for the costs that determine the rewards. To find good policies for the non-stationary stochastic lot-sizing problem, using an appropriate demand distribution for generating the demand samples is crucial. There are several studies looking into stochastic inventory problems with non-stationary and stochastic demand. In the majority of the studies, the parameters of the demand distribution in each upcoming period are known, and typically considered to be independent random variables (Ma et al., 2022; Amiri et al., 2023; Dehaybe et al., 2024) . To the best of our knowledge, no multi-item inventory studies have been conducted where demand satisfies realistic properties such as being autoregressive, integer and non-negative. van Hezewijk et al. (2023a) present a non-stationary demand generating process (DGP) that satisfies several realistic properties: having non-negative, integer and autoregressive demand. In this paper, we use that DGP to generate realistic demand samples to be used in DRL, and validate that DRL can be used to find good policies for problems with realistic demand processes.

The second challenge of applying DRL for this problem is how to deal with (non-stationary) uncertainty. Dulac-Arnold et al. (2021) document uncertainty as one of the main challenges of real-world DRL. In case of a highly stochastic environment, differentiation between good and near-optimal actions is difficult due to the high variance encountered in training. Temizöz et al. (2023) propose an algorithm called Deep Controlled Learning (DCL) which uses a variance control mechanism to speed up and improve the learning in stochastic and stationary environments (i.e., the class of MDPs with Exogenous Inputs, MDP-EI). In this paper, we validate the potential of using the DCL algorithm in non-stationary environments. We also demonstrate that by including the forecast information into the state information, the trained model can continue to be used when demand changes.

Third and most importantly, it is challenging to handle the exploding action space that results from combinatorial optimization problems (Boute et al., 2021) . Several approaches for dealing with the action space explosion have been proposed in literature. In this paper, we propose to decompose the problem: we provide an equivalent and non-combinatorial formulation of the Markov Decision Process (MDP) by decomposing the full period decision into sub-decisions. We then proceed to demonstrate that this decomposition allows us to solve larger and more complicated problems. To this end, we adopt as benchmark policy the tailored aggregate modified base-stock (AMBS) heuristic introduced by van Hezewijk et al. ( 2023b), and we refine this benchmark by relaxing the batch restrictions imposed in van Hezewijk et al. (2023b) , in line with our problem definition. We demonstrate that the DRL method with action decomposition proposed in this paper is superior to the refined AMBS heuristic for problems with up to 15 products, whereas the Proximal Policy Optimization (PPO) implementation by van Hezewijk et al. (2023b) can beat the refined AMBS benchmark for problems of up to 5 products.

This paper integrates a realistic auto-regressive demand process, the DCL algorithm and a decomposition approach to solve the nonstationary SCLSP. The DRL algorithm proposed in this paper is demonstrated to find good policies for multi-item production and inventory systems with setups and non-stationary demand. Our findings confirm that using the proposed method enables us to successfully extend the application of DRL to more realistic problems. The remainder of the paper is structured as follows: first an overview of existing literature is provided in Section 2. Then, the DGP, the non-stationary SCLSP and its decomposition are presented in Section 3. The DCL methodology and benchmarks are explained in Section 4 and we examine the performance of this method and decomposition approach in Section 5. We conclude the paper in Section 6.


## 2. Literature review

Our work contributes to better solving multi-item inventory problems, and to modeling and solving inventory problems with nonstationary demand. Accordingly, we review the work on DRL for problems that have similar characteristics to the multi-item SCLSP with non-stationary demand. We begin with providing an overview of general research on inventory control with non-stationary demand. We then present work related to solving inventory problems with stochastic non-stationary demand using DRL. Finally, we analyze the literature on handling combinatorial action spaces in DRL.

Inventory control with non-stationary demand: For modeling the product demand in our multi-product inventory model, we adopt a demand process proposed by van Hezewijk et al. (2023a) that yields discrete and non-negative demands to stay close to practice, and that is otherwise similar to the widely adopted ARIMA-type processes with normally distributed errors. Some researchers have investigated inventory control when demand follows such ARIMA-type processes in single-item settings. In particular, Graves (1999) describes the demand using a (0,1,1) IMA process, wherein there is a periodic adjustment to the demand mean, adjusted by a factor 𝛼 relative to the magnitude of a normally distributed shock with mean 0 and variance 𝜎 2 . Graves proceeds by introducing a modification to the safety stock formula that depends on lead-time (𝐿) and 𝛼, and asserts that we require 'dramatically more safety stock when demand is non-stationary, in comparison with the textbook case of stationary demand' (p. 54). Strijbosch et al. (2011) adopt a (0,1,1) ARIMA model for demand and modify it by truncating the mean to avoid negative values. They find that the key to improving stock control efficacy lies more in updating the demand variance accurately than in selecting specific forecasting techniques or parameters. Prak and Teunter (2019) develop a methodology to integrate uncertainty in future demand into inventory models, employing models of demand that assume trends and that involve random walks. Some research addresses the prediction of lead-time demand distributions in the supply chain setting: Cao and Shen (2019) suggest using a neural network to predict demand quantiles for nonnegative, non-stationary autoregressive demand processes. Babai et al. (2022) investigate three methods for estimating the variance of leadtime demand and present analytical results for an ARMA(1,1) demand model. Song and Zipkin (1993) introduces an alternative approach to modeling non-stationarity: They adopt a continuous-time Markov chain, where demand in state 𝑖 follows a Poisson process with rate 𝜆 𝑖 . Subsequent research by Bayraktar and Ludkovski (2010) and Hu et al. (2016) expands on these Markov-modulated demand processes by examining the impacts of partial demand observability and demand correlations on optimal policy structures, respectively. A common finding is that optimal policies tend to be of state-dependent (𝑠, 𝑆) type. Determining such parameters can be complex and analytically challenging.

DRL for inventory control with stochastic and non-stationary demand: There exist a large variety of DRL algorithms for approximately solving MDPs (Shakya et al., 2023) , most of which are designed for and tested in deterministic environments. When there is only one deterministic consequence of an action, it is relatively straightforward to evaluate the performance of an action based on one single observation. However, when there are multiple possible consequences (rewards) of an action in a stochastic environment, it is more difficult to assess the value of different actions in a state based only on one observation of the reward. Temizöz et al. (2023) present the DCL algorithm, which uses simulation techniques like Common Random Numbers to control the variance of these value approximations, leading to significantly better performance in stochastic environments. They explore the class of MDPs with Exogenous Inputs (MDP-EI), where the Exogenous Inputs (demand) are assumed to be independent of the states and actions. In this paper, we demonstrate the application of the DCL algorithm to non-stationary environments, by training the algorithm directly for such an environment.

The case of non-stationary demand is increasingly studied in the DRL for inventory control domain. Dehaybe et al. (2024) take nonstationary environments into account by explicitly embedding the forecasted parameters of the demand distribution in the state space. They use the PPO algorithm with a continuous action space to find a good policy in a single-item lot-sizing problem. By training the neural networks with a large variety of forecast information (seasonality, linear growth and decline, etc.), the trained networks can be reused. Stranieri et al. (2024) also consider seasonality in a two-echelon supply chain, and combine reinforcement learning with two-stage stochastic programming. van Dijck et al. ( 2024) consider a single-product multiechelon assembly system with non-stationary demand, and benchmark analytical and DRL policies based on case studies arising in the hightech industry. By developing analytical policies tailored to the problem, they can vastly outperform classical benchmarks and come relatively close to the performance of DRL. Amiri et al. (2023) propose an Adaptive Stochastic Convex Bandit algorithm that, in addition to optimizing the exploration-exploitation trade-off that is present in DRL, also considers the balance between keeping and forgetting information. The algorithm learns to detect changes in the environment and uses this as a signal that more exploration would be needed again as the old information becomes outdated. This algorithm outperforms other DRL algorithms, and shows that having some mechanism to handle new information is quite relevant. Temizöz et al. (2024) consider DRL methods that deal with situations where problem parameters, such as cost parameters and demand and leadtime distributions, are uncertain. They propose the train, then estimate and decide (TED) framework that enables DRL policies to be applied directly to settings with unknown parameters. They apply this framework to single-item inventory control problems with unknown demand and demand censoring, and find that it outperforms policies tailored to such settings.

We are not aware of any studies considering multi-product inventory problems with non-stationary demand. Moreover, unlike the approaches proposed in prior work, we propose to directly train our DRL policy on an generative ARIMA type demand model, which enables the policy to learn to account for leadtime demand uncertainty associated with forecast inaccuracy and auto-correlated demand.


## Combinatorial action spaces in DRL:

A key concern when applying DRL algorithms in practice is the ability to solve large problems. While there seems to be remarkable performance of neural networks in terms of processing and generalizing learnings from multi-dimensional inputs (Boute et al., 2021) , the performance of DRL in multi-action problems is still under investigation. Multi-item lot-sizing problems require an action for every item. Several applications of DRL in such problems assign one output node of the neural network to each combination of actions, meaning that the action space grows exponentially with the number of stock points or items considered in the problem (Vanvuchelen et al., 2020; van Hezewijk et al., 2023b) . This large action space dramatically reduces the performance of the DRL algorithm.

To overcome this issue, Kaynov et al. (2024) present a multi-discrete action distribution which shows only a linear growth in the action space. They consider a divergent multi-echelon inventory problem, where the neural network outputs several discrete probability distributions, one for each stock point. They use the PPO algorithm to find good policies, but they encounter decreasing performance in case of an increasing number of stock points. As there are no longer a priori guarantees that the actions placed by the different stock points are feasible, a random sequential allocation rule is implemented to steer the DRL algorithm towards learning feasible combinations of actions.

Another approach for dealing with large action spaces is to use continuous actions. Geevers et al. (2023) study general multi-echelon inventory problems, where the neural network outputs a continuous probability distribution for each stock point. This reduces the size of the action space dramatically. Vanvuchelen et al. (2025) study a joint inventory replenishment problem using a continuous action space. They demonstrate that this scales well to large problems. In both studies, a mapping function was designed to map the continuous actions to feasible discrete actions.

We contribute to literature by demonstrating an approach for solving large non-stationary lot sizing problems, bringing us a step closer to solving realistic versions of the multi-item SCLSP. To successfully learn DRL policies that effectively deal with non-stationary demand, we include forecast information in the MDP state space. As a consequence, there is no need for retraining the DRL agent in case of changing demand. To address the large actions space, we propose to decompose the problem into discrete sub-decisions, in order to retain the resource allocation decision for the DRL agent. This precludes the need to design a mapping function, which is advantageous as designing a mapping function for our problem is non-trivial because of the setup costs.


## 3.1. Problem description and notation

We study a stochastic capacitated lot sizing problem (SCLSP), where multiple products (𝑖 ∈ 𝐾) are produced on a single machine with a limited production capacity ( C). Time is divided into periods 𝑡 of equal length (e.g. weeks). Production is capacitated, meaning for every period 𝑡, a certain amount of production time is available (e.g. 40 h). Producing a product takes a certain amount of time (𝑞 𝑖,𝑡 ), and switching over from producing a certain product 𝑗 to another product 𝑖 comes with a setup time (𝜃 𝑖 , e.g. due to cleaning or recalibrating) and costs (𝑘 𝑖 ). Setup times and costs are assumed to be independent of the sequence. For every period 𝑡, we need to determine what quantity will be produced for every product (𝑞 𝑖,𝑡 ). At the end of a period, after all production has taken place, the demand for the different products is observed (𝑑 𝑖,𝑡 ), and satisfied directly from stock if possible. Unsatisfied demand is backordered.

The SCLSP is identical to the one described by van Hezewijk et al. (2023b) . This problem is modeled as a Markov Decision Process (MDP), requiring a definition of the states, actions, rewards and transitions. In the study by van Hezewijk et al. (2023b) , all production quantities for the full period were determined simultaneously, and the demand was considered stationary. These dynamics are shown in Fig. 1 .

In this paper, we propose a new MDP formulation that contains multiple sub-decisions in each period, and we consider stationary as well as non-stationary demand. The dynamics of both formulations are equivalent, the only difference is that the production sequence is an explicit decision in the decomposed reformulation. The notation for this new MDP formulation is summarized in Table 1 . To create a production plan, the time in one period 𝑡 is split up in small discrete time units (e.g. minutes or seconds), and 𝜏 indicates how many time units are already used. At time 𝜏, first we take the 'first-stage' subdecision of which product to produce, potentially incurring setup costs. Then we take the 'second-stage' sub-decision of how much of that product to produce. These sub-decisions are taken over and over, until either (1) all production capacity is consumed, or (2) we decide to stop production before consuming all capacity. When one of these conditions apply, the created production plan is implemented for period 𝑡. At the end of period 𝑡, the demand is observed and holding and shortage costs are incurred. The new MDP formulation is shown in Fig. 2 , where the sub-decisions and transitions within the period are connected by dashed arrows. Note that our model assumes that demand is released in its entirety at the end of the period. Given that the production schedule is composed of sub-decisions, an alternative formulation could reveal demand throughout the period. This would enable the production subdecisions in later periods to be dependent upon demand revealed while implementing earlier sub-decisions, i.e. while making the first batches. This alternative formulation is not pursued in this paper for simplicity, and to keep the problem comparable to the problem studied by van Hezewijk et al. (2023b).


## 3.2. State space

The state of the MDP satisfies the Markov property: the future actions only depend on the current state and not on past history. The state space (1) contains information about the inventory position of product 𝑖 (𝐼 𝑖,𝑡 ), the forecasted mean and standard deviation of demand for product 𝑖 in period 𝑡 (𝜇 𝑖,𝑡 , 𝜎 𝑖,𝑡 ), a variable that indicates which product is setup on the machine (𝜔 𝑖 ), the amount of capacity that is already used in the period (𝜏) and the amount of units that have already been produced for product 𝑖 in period 𝑡 (𝑞 𝑖,𝑡 ).

𝑠 𝜏 = {𝐼 𝑖,𝑡 , 𝜇 𝑖,𝑡 , 𝜎 𝑖,𝑡 , 𝜔 𝑖 , 𝜏, 𝑞 𝑖,𝑡 } ∀𝑖 ∈ 𝐾 (1)

In the beginning of the period, 𝑞 𝑖,𝑡 = 0 ∀𝑖 ∈ 𝐾. Typically, 𝜏 = 0 as well, unless a setup from the previous period needs to be finished (see Fig. 2 ). Due to the sub-decisions being taken within one period, the state variables 𝜔 𝑖 , 𝜏, 𝑞 𝑖,𝑡 are updated during the period. The other variables 𝐼 𝑖,𝑡 , 𝜇 𝑖,𝑡 , 𝜎 𝑖,𝑡 are only updated at the end of the period.


## 3.3. Action space

Based on the information in the state space, an action is taken, either a first-stage or a second stage action. A 'first-stage' action 𝐴1 𝜏 determines which product to produce:

𝐴 1 𝜏 ∈  1 , where  1 = {𝑝 0 , 𝑝 1 , … , 𝑝 𝐾 } (2)

The action '𝑝 0 ' represents that production is stopped and the machine will go idle until the next period. The 'second-stage' action 𝐴 2 𝜏 determines the production quantity for the product that will be produced. This production quantity is limited by the remaining available capacity:

𝐴 2 𝜏 ∈  2 𝜏 , where  2 = {𝑞 1 , … , 𝑞 C-𝜏 } (3)

Starting in the new period, the machine can continue producing the product that was produced last in the preceding period without needing a setup. Note that in theory it is possible to select the same product multiple times to be produced next in one period. However, as no additional information becomes available during the period, selecting the same product multiple times does not provide benefits and is therefore not allowed in the MDP.


## 3.4. Transitions and demand process

The transition dynamics are shown in Fig. 2 . During the period, the transitions to new states are deterministic, i.e. we know exactly in which state we will end up after taking an action, as there is no uncertainty in production yield or setup times. However, in moving from one period to the next, the transition becomes stochastic as the demand that is observed is drawn from stochastic distributions. This stochastic distribution could be stationary, but it could also be that a new forecast for the demand and standard deviation becomes available every period. The stochastic distributions that are studied in this paper are described in more detail in the experimental setup in Section 5. If there is an incomplete setup in a period, the costs for the setup are incurred in the period where the setup is started.


## 3.5. Rewards

In this SCLSP, there are holding costs, backorder costs and setup costs. The period costs are calculated following Eq. (4).

> **Eq. (4) — 기간 비용 (PDF에서 이미지로 추출 실패, 본문·notation 기반 재구성).**
> 기간 $t$의 비용 = 보유비 + 백오더비 + 셋업비의 제품 합산:
>
> $$C_t = \sum_{i \in K}\left( h_i\,[I_{i,t}]^+ + b_i\,[I_{i,t}]^- + k_i\,z_{i,t} \right)$$
>
> $[I_{i,t}]^+ = \max(I_{i,t},0)$ (보유 재고), $[I_{i,t}]^- = -\min(I_{i,t},0)$ (백오더), $z_{i,t}\in\{0,1\}$ 셋업 지시자.
> *주의: 위 형태는 verbatim 아님 — 원문 Eq.(4) 이미지 확인 필요.*

The goal in the MDP is to find a good 'policy' for the decision maker: a function 𝜋 that specifies the action 𝑎 𝜏 that the decision maker will choose when in state 𝑠 𝜏 . In theory, we could find the optimal policy 𝜋 * (𝑠 𝜏 ) by solving the recursive Bellman Optimality Equations, but this quickly becomes computationally prohibitive. Therefore, we resort to approximate solution methods, further described in the next section.


## 4. Methodology

To solve the new decomposed MDP formulation, we select a DRL algorithm, described in Section 4.1. We describe the rollout policy, a key element of the selected DRL algorithm, in Section 4.2, and elaborate on the state and action representation in Section 4.3 and Section 4.4.


## 4.1. DRL algorithm selection

As shown in Section 2, the PPO algorithm is a very popular DRL algorithm in the operations management domain. Also, as PPO was used to solve the full period formulation of the SCLSP problem, it would be interesting to investigate how PPO would perform when solving the new decomposed MDP formulation. However, in conventional implementations of PPO (using the gymnasium API 1 ), each action transition equals taking one time step. However, in this decomposed MDP formulation, multiple actions can take place in one time step. When each action is equal to one time step, we expect that the actions that are taken before the end of the period seem very favorable. Only at the end of a period, as demand is observed, the holding and backorder costs are incurred. An agent would learn to postpone the end of a period by taking more actions, so producing many different products in small quantities.

Preliminary experimental results have supported this conjecture. When attempting to solve the new MDP formulation with PPO, the PPO algorithm is found to converge to a solution where on average 7.4 actions are taken per period. However, from the parameters of the AMBS heuristic, our benchmark solution, we know that this kind of policy is not desirable. In fact, considering a problem with 𝐾 = 5 products, the AMBS heuristic would suggest to only allow one new setup per period, which would typically require only 3 actions (setup product 𝑖, decide quantity 𝑞 𝑖,𝑡 and go idle). As a consequence, the costs of the policy trained via PPO are 3.6 times higher than the costs of the AMBS heuristic. PPO is broadly considered to be among the best-performing model-free algorithms for inventory control (Vanvuchelen et al., 2025) , and is widely adopted for that purpose (see also Geevers et al., 2023; Dehaybe et al., 2024; Kaynov et al., 2024; Stranieri et al., 2024) . Since PPO fails to yield good policies for our decomposed problem, we are inclined to adopt a rather different DRL algorithm for finding policies in the decomposed MDP proposed in this paper. Indeed, since other model-free DRL models (e.g. A3C Gijsbrechts et al., 2022) rely on the same trace data, they can be expected to run into similar difficulties. Also, A3C appears to require extensive hyperparameter tuning and is reported to be outperformed by the best-performing benchmarks for the standard lost sales inventory problem (Gijsbrechts et al., 2022) .

We propose to adopt the deep controlled learning (DCL) algorithm: unlike PPO and A3C, it was demonstrated to outperform all state-ofthe-art benchmarks for a range of simple inventory problems, including the lost sales inventory control problem and perishable inventory systems (Temizöz et al., 2023) . Moreover, standard DCL implementations exist that correctly account for cases with multiple sub-actions in a single time step (Akkerman et al., 2023) . In the DCL algorithm, several approximate policy improvement steps are executed to improve the policy for the MDP. A neural network is used as the policy approximator, taking the state as an input, and giving the action to take as the output. Instead of learning the policy parameters using policy gradient methods, in which we train the network by optimizing the parameters with respect to the expected return of the actions, DCL uses supervised learning to classify the best action in a state. By simulation, we determine what the best action is for each sample state. The neural network in DCL is trained to identify the correct action given the labeled data, without considering the reward associated with the actions (see Temizöz et al., 2023 , for details on neural network training). A dataset (of size 𝑁) that is used to train this network is constructed in the first generation by following the rollout policy. The selection of the rollout policy is described in Section 4.2. Per sampled state, we rollout each action 𝑀 times for 𝐻 periods to estimate the action-value function. Then for all 𝑁 sampled states, we know the best action to take. To extend the learnings to the entire state space, a neural network is used to approximate the policy beyond the sampled states. In subsequent generations, the neural network policy is used as the rollout policy.

The DCL algorithm uses techniques from simulation to efficiently learn good policies. For a complete description, see Temizöz et al. (2023) . Here we highlight two key features of the algorithm. The first key element is that Common Random Numbers are used to reduce the variance in action-value estimates. This means that the same observations of the uncertain variable are used for evaluating the different actions, removing a lot of noise from the estimates. The second key component is that the DCL algorithm allocates the rollout budget for a state (𝑀 ⋅ |𝐴 𝑠 |) using Sequential Halving, a technique that allocates more rollouts to promising actions, preventing the unnecessary usage of computational resources on actions that are suboptimal. After performing 𝑟 rollouts for all actions, the worst half is dismissed for further analysis. For the remaining actions, more rollouts are performed to find more accurate estimates, again dismissing the worst half after 𝑟 rollouts. This continues until there is one action remaining. In Section 5.2 we describe the hyperparameters of the DCL algorithm that were used. Whereas Temizöz et al. (2023) use state-independent random events, in our setting, the events are dependent on the forecasted mean and forecasted deviation, which are part of the state space (1).


## 4.2. Rollout policy

In the first generation of the DCL algorithm, all actions are compared with each other, based on the rewards of taking the action 𝑎 𝜏 and taking 𝐻 subsequent steps of the rollout policy. The DCL algorithm can benefit significantly from selecting an appropriate policy to be the initial rollout policy (Verleijsdonk et al., 2024) .

For the SCLSP, we consider the AMBS heuristic as described by van Hezewijk et al. (2023b) as the best performing heuristic. However, this heuristic takes a full period decision and can therefore not be easily used as a rollout policy for this problem with sub-decisions in the period. To define a rollout policy, we use the ideas behind the AMBS heuristic and adjust them to work in the new MDP formulation with sub-decisions.

At the start of a period 𝑡, we perform a setup for the product 𝑖 that has the largest expected gap to the reorder level (calculated using 𝐵 𝑚𝑖𝑛 ). For this product 𝑖, we will attempt to produce the quantity needed to reach the order up to level (calculated with 𝐻 𝑚𝑎𝑥 ), of course limited by the capacity available. After production, if there is still capacity remaining and if we are below our setup quota (𝑍 𝑚𝑎𝑥 ), we start producing the product 𝑗 with the second largest gap to the reorder level. In the case that there is no product that has a gap to the reorder level, we produce the product that is currently set up to the order up to level. If no more production is allowed due to reaching the setup quota and order up to levels, production is stopped and the machine goes idle for the remainder of the period. The technical description of the rollout policy is given in Algorithm 1. 𝐵 𝑚𝑖𝑛 , 𝐻 𝑚𝑎𝑥 and 𝑍 𝑚𝑎𝑥 are parameters found by searching a grid and evaluating the performance using simulation. The product that is set up (i.e., 𝑖 for which 𝜔 𝑖 = 1) is denoted by 𝜔.


## 4.3. State representation

We represent the inventory policy for our decomposed MDP as a neural network. In particular, we adopt a dense (fully connected) neural network that takes as input a vector representation of the state (cf. Boute et al., 2021) , and that has one output for every possible action in the decomposed MDP.

The hidden layers support the representation of a vast range of possible relations between the state and the action. Nevertheless, having a smart representation of state information can reduce the number of nodes and/or layers needed in the neural network, hence speeding up the learning. In the SCLSP, one of the most important pieces of information, both in the first and second stage action, is the inventory, demand and standard deviation forecast of the product that is currently set up (𝜔). While the product that is set up can be identified by the value of 𝜔 𝑖 , presenting the information for that product in a consistent place in the vector representation of the state could be supportive of better learning. Therefore, the representation for the product that is currently set up is presented first in the vector, followed by information In cases with stationary demand, we consider an infinite horizon MDP.

In cases with non-stationary demand, we consider a finite MDP, with at most 100 periods. The last element of the state space is the number of remaining periods, and it is only included in the finite horizon MDP.


## 4.4. Action representation

The definitions of the first and second stage action are given in (2) and (3). The second stage decision is to decide on the production quantity, which takes the available capacity as a maximum. So in case there are 100 units of capacity available, there are 100 actions. However, one could imagine that it is not that interesting to analyze the difference between quantity 91 and 92, while the difference between quantity 1 and 2 is quite relevant. So there is potential to reduce the action space by focusing on the relevant quantities in the action space. Therefore, we define the restricted second stage action space:

 2 * 𝜏 = {𝑞 1 , 𝑞 2 , … , 𝑞 20 , 𝑞 22 , … , 𝑞 40 , 𝑞 45 , … , 𝑞 60 , 𝑞 70 , … , 𝑞 100 , 𝑞 120 , … , 𝑞 200 , 𝑞 300 , … , 𝑞 C-𝜏 } (6)

In the DCL algorithm, we use one action space for the neural network that combines any possible action in any of the MDP states (see Temizöz et al., 2023) . Depending on the type of decision (first or second stage) to make, the remaining capacity C -𝜏, and the product currently set up, part of action space is masked out. Thus, the action space in the DCL algorithm is defined as:

 =  1 ∪  2 * 0 (7)


## 5. Results

We study the performance of the DCL algorithm in a variety of settings. We first compare the results of the DCL algorithm to the PPO The experimental setup of the non-stationary experiments is described in Section 5.2, and their results are presented in Section 5.3. We discuss computation times in Section 5.4.


## 5.1. PPO comparison in stationary settings

In van Hezewijk et al. ( 2023b), the SCLSP is modeled as an MDP following the transition dynamics in Fig. 1 . Every period, the production quantities for all products are decided in one full period decision. This means that the number of potential actions increases exponentially in the number of products. Action space reduction and masking techniques are added to the PPO algorithm to allow the PPO algorithm to outperform the benchmark. Additionally, the decision granularity is reduced by restricting production quantities to be a multiple of a fixed batch size, reducing the flexibility in decision making. For example, the batch size would be 2.5 times the average period demand for a product in case of experiments with 10 products. Despite these modifications, the applicability of the PPO algorithm seems to be limited to problems with up to 10 products.

In this section, we compare the performance of the DCL algorithm for solving the new decomposed formulation of the SCLSP with subdecisions with the performance of the PPO algorithm in the MDP formulation of the SCLSP with full period decisions. As our goal is to arrive at a fair comparison, we adopt experimental settings of van Hezewijk et al. (2023b) , in particular the settings corresponding to parameter set 1 in Table 2 of van Hezewijk et al. (2023b) . We focus on cases with 5 and 10 products, and in addition we investigate the case of 15 products that yielded an intractable action space with the full period decision model proposed in van Hezewijk et al. (2023b) . All other problem parameters are also consistent with the experiment for parameter Set 1: We let 𝑏 𝑖 = 9, ℎ 𝑖 = 1, 𝑘 𝑖 = 200 and 𝜃 𝑖 = 0; we investigate two capacity levels (1.1 and 1.5), where the total available capacity is calculated by multiplying the capacity factor with the total expected initial demand: C = 𝑓 𝑐 ⋅ ∑ 𝑖∈𝐾 𝜇 𝑖 . Finally, demand follows a discrete uniform distribution, with two possible distributions: 𝑈 {3, 5} and 𝑈 {0, 8}.

We adopt the DCL implementation from the DynaPlex DRL library (Akkerman et al., 2023) . The hyperparameters for the DCL algorithm are shown in Table 2 ; for other hyperparameters, the neural network training procedure, and the loss function, we follow Temizöz et al. (2023) . To ensure that our results can be replicated, the environment and training code have been made available as examples in the DynaPlex library.

The performance of the DCL algorithm is determined by using the best trained network as a policy in 10,000 simulation runs with a length of 100 periods per run. This is compared with the performance of the AMBS heuristic and rollout policy. All policies are encountering the same demand sequences, that are generated independently from any sequences used for training, in order to enable a fair comparison.

The results of the comparison between the performance of PPO in the full period formulation of the MDP and DCL in the new decomposed MDP formulation are shown in Table 3 . The DCL algorithm outperforms the PPO algorithm and the AMBS heuristic, also in cases with higher levels of uncertainty. Additionally, the DCL algorithm is able to outperform the AMBS heuristic in problems with 15 products. For settings with ≥20 products, we found the DCL algorithm to perform roughly on par with the AMBS heuristic. By increasing the sample size and neural network (more layers and/or more nodes in a layer) the performance of the DCL algorithm improved somewhat, but these strategies appear to be insufficient to clearly outperform the AMBS heuristic in line with the performance benefits for 10 and 15 products. In Section 6 we discuss ideas for overcoming this limitation. We expect that in these cases, more benefits are to be gained from a specialized neural network architecture, where there is a clearer distinction between the two different actions that the agent is trying to learn.

Note that the performance gap of PPO to AMBS is not identical to the results presented by van Hezewijk et al. (2023b) : In that paper the AMBS heuristic was impaired by restricting it to produce multiples of a fixed batch size, whereas in this paper we do not consider such a restriction. That means that the AMBS performance in this paper is better than in van Hezewijk et al. (2023b).


## 5.2. Experimental setup for non-stationary settings

In the non-stationary settings, we are interested to see how the DCL algorithm is impacted by uncertainty, non-stationarity, and to see how the MDP with sub-decisions scales to larger problem settings. In the experiments, the DGP of van Hezewijk et al. (2023a) is used. Three key parameters are needed to use that distribution: initial expectation 𝜇 0 , initial standard deviation 𝜎 0 and smoothing parameter 𝛼.

To be able to select the right levels for the coefficient of variation (𝐶𝑂𝑉 ) and 𝛼 for the experiments, we need to consider the interplay between the non-stationarity and available capacity. In stationary settings, the utilization rate is always lower than 100%, indicating that the available capacity is higher than the expected demand. In the nonstationary cases this is also the case for the initial expected demand, but as the demand distributions for the products change over time, we run the risk that this available capacity turns out to be insufficient to meet expected demand. In the experiments we want to study problems where no strategic capacity interventions are needed. In Section 5.2.1 we demonstrate the variability and non-stationarity settings for which strategic capacity interventions are required. In Section 5.2.2 we provide an overview of the parameter settings that are studied in this paper.


## 5.2.1. Risk of capacity shortage

As van Hezewijk et al. (2023a) show, the DGP will generate samples that are unbiased, and the expectation of the long-term demand over the samples remains similar to the initial forecast. However, when considering problems with a small number of products (i.e., taking a very small sample of the DGP), this property will not always be visible. Fig. 3 shows the number of times where the total daily demand exceeds the daily capacity (i.e., daily capacity shortage). We see that with a lower number of products, an increasing 𝐶𝑂𝑉 and increasing 𝛼, the number of periods in a trajectory where the total daily demand exceeds the daily capacity increases. Occasional capacity shortages are not problematic, as the cumulative capacity over the whole trajectory can be used to build inventory to cover demand in all periods. However, a high number of daily capacity shortages can indicate long-term capacity issues. In the analysis, we ignore the presence of setup times.

In case of setup times (𝜃 𝑖 > 0), encountering capacity shortages is even more likely. Fig. 4 shows the impact of 𝐾, 𝐶𝑂𝑉 and 𝛼 on the probability that the cumulative demand over a time horizon of length 𝑇 exceeds the cumulative capacity over this horizon. Indeed we see that the cases for which we observe a high number of daily capacity shortages also result in long-term capacity shortage. Very high backorder costs are a result of a long-term capacity shortage. In practice, strategic interventions to increase or decrease production capacity based on expected changes in demand are very reasonable, but they are beyond the scope of this paper.


## 5.2.2. Experiment parameters

Problem instances with an increasing number of products 𝐾 are considered (5, 10, 15) , as this has an impact on the size of the action space. Additionally, the capacity has an impact on the size of the action space, so this is also taking multiple values in the expe riments. The total available capacity is calculated by multiplying the capacity factor with the total expected initial demand: C = 𝑓 𝑐 ⋅ ∑ 𝑖∈𝐾 𝜇 𝑖,0 . We consider two factors for 𝑓 𝑐 (1.5, 2.5). Based on the analysis from Section 5.2.1, we select a low and high value for 𝐶𝑂𝑉 (0.5 and 1.0), and three parameters for 𝛼 (0.0, 0.025, 0.05). With these settings, the need for strategic interventions in the production capacity is limited.

The other relevant problem parameters for costs and setup time are kept constant across the experiments. We draw the readers attention to the effects of the non-stationary demand on the setup costs. The setup costs were determined based on the expected order frequency, or time between orders (TBO; 𝑘 𝑖 = ℎ 𝑖 ⋅𝑇 𝐵𝑂 2 ⋅𝜇 𝑖 2

). However, as the mean may change over time, either the TBO or setup costs could change. We have chosen to keep the costs constant over time, but this of course means that the initial expectation of the frequency of orders could change as the mean changes.

The different experimental settings result in a total of 36 experiments to be carried out. A summary of the settings can be seen in Table 4 . We use the same hyperparameters for the DCL algorithm in both the stationary and non-stationary experiments, as shown in Table 2 . 


## 5.3. Experimental results for non-stationary settings

In Table 5 the results for the non-stationary experiments are shown. We observe that the DCL policy consistently outperforms the AMBS policy.

For small problems (𝐾 = 5), the gap of the DCL to the AMBS policy increases as the capacity increases. However, for larger problems we observe a reverse effect. There could be multiple reasons for this. First of all, as the number of products and capacity increases, the likelihood that too many products are competing for the same resource simultaneously decreases, and there is more 'risk pooling' of the different products. In such cases, the AMBS heuristic might in general be closer to the optimal solution, and is harder to outperform with large gaps. Second, in these cases the action space is quite large, and we might once again observe the detrimental impact of a large action space on the performance of DRL. Generally, we also see that in cases of a larger COV, the gap between the AMBS and DCL heuristic becomes slightly smaller. Fig. 5 shows a breakdown of the AMBS and DCL policies in terms of the inventory levels, setups and backorders. Generally, the DCL policy results in lower inventory levels than the AMBS policy, except for cases with a high 𝛼 and tight capacity. With higher levels of non-stationarity, or tighter capacity, more inventory is needed.

The DCL policy also results in a lower number of setups per period than the AMBS policy. As the level of non-stationarity or uncertainty increases, the number of setups decreases. Instead of having more flexibility in production due to the non-stationarity or uncertainty (i.e., more setups), production seems to become less flexible. However, larger batches are produced, which results in longer intervals between two setups of the same product. With a larger capacity, less setups are required.

The DCL policy results in lower backorder levels than the AMBS heuristic. As expected, with higher non-stationarity and uncertainty in the demand, the number of shortages increases dramatically. Having larger capacity available results in lower backorder levels, signaling the importance of adjusting the production capacity in case of non-stationary demand.


## 5.4. Computation times

The computation times for the DCL algorithm are shown in Table 6 ; we report the cumulative time spent collecting samples and training neural networks over the course of training 5 generations of neural networks. Sample collection took place on 4 parallel nodes, each equipped with a AMD Genoa 9654 processor. While training the DCL algorithm is slower than finding parameters for the AMBS heuristic by grid search, the DCL algorithm is faster than DRL algorithms that require extensive hyperparameter tuning: it is more than twice as fast as the best-performing PPO algorithm implementation of van Hezewijk et al. (2023b) , while substantially outperforming that policy. The majority of the computation time can be attributed to the sample collection that determines the value of actions. The number of products is the main determinant of the computation time, while the values of 𝛼, 𝑓 𝑐 and the 𝐶𝑂𝑉 have limited impact.


## 6. Outlook and conclusion

We investigated the non-stationary SCLSP where the demand follows an autoregressive demand process, and the aim is to minimize setup, holding and backorder costs. This problem is notoriously impacted by the curse of dimensionality as the number of products under consideration increases. To address this challenge, we model the problem as an MDP and decompose the full period decision into subdecisions. We also study whether the strengths of the DCL algorithm for finding good policies in stationary stochastic problems transfer to non-stationary settings. We find that the DCL algorithm consistently finds policies that outperform the benchmark for problems of up to 15 products, both under stationary and non-stationary demand processes. The decomposition of the decisions in a period allows us to solve larger problems compared to prior work (van Hezewijk et al., 2023b) , and thus it appears to be a very promising mechanism for reducing the action space in combinatorial problems. Furthermore, by including demand information in the state space, the trained model can be used even as demand changes, thus avoiding the need to retrain.

Even with the action space decomposition, the algorithms fails to clearly outperform the benchmark when there are 20 products. This may very well be due to a failure of the neural network to efficiently learn and generalize in large-scale settings. In particular, note that the neural network receives as input a vector that lists the state representation for all products (see Section 4.3), and needs to somehow identify the optimal product to be produced next. Within the dense neural network architecture, this can only be achieved by independently evaluating the status of each individual product, which inherently limits scalability. Recent work (e.g. van der Haar et al., 2024) has explored global learning for inventory control in settings where a policy must be trained to control inventory for a large number of SKUs independently. In particular, when setting the action for a specific product, only the representation of that specific product is input to neural network that shares parameters over all products. This enables the learning of a single set of neural network parameters that will be applicable for all products, which effectively supports parameter sharing across products. The resulting method is reported to scale to 10,000 products (van der Haar et al., 2024) . This approach is not applicable in our setting: note that the decision which product to produce next cannot be made based on the state representation of a single product. Thus, while harnessing global learning and/or parameter sharing could substantially increase scalability for our problem, achieving it is non-trivial in our setting, especially since our problem involves both the selection of the next product to set up, and the quantity decisions.

The non-stationary demand process adopted in our model is autoregressive; it is a stochastic process that takes on discrete, non-negative values and that is otherwise very closely related to the celebrated ARIMA(0,1,1) process, which is appropriate for modeling demand for a broad range of products, see van Hezewijk et al. (2023a) . To apply the methods developed in this paper for products that exhibit clear seasonality and or trends, we would need to move beyond ARIMA(0,1,1) models towards models that exhibit such demand patterns. It is well-known that also for such models there exist generative models, i.e. procedures that yield demand sequences with the desired properties (Hyndman et al., 2002) , which should provide an avenue towards training policies for our inventory problem that perform well for products with demand that exhibits seasonal or trended patterns. Note however, that Hyndman et al. (2002) and related state space models tend to use Gaussian error terms and thus negative values that need to be dealt with somehow.

A final direction for future work is extending the current model to accommodate multiple machines, which are common in real-world industrial settings. One approach could involve a decision loop for each product, where the model determines whether to produce the item, selects the most appropriate machine, and then decides on the production quantity. Alternatively, each machine might decide on the product to produce next whenever it completes production, in a fashion that is closely related to the present paper. Exploring these ideas would not only enhance the applicability of the model but also contribute to the optimization of complex manufacturing systems. 

International Journal of Production Economics 284 (2025) 109601

https://gymnasium.farama.org/


## References

- N Amiri, M Udenio, R Boute (2023). Adaptive multi-armed bandits for non-stationary inventory control
- M Babai, Y Dai, Q Li, A Syntetos, X Wang (2022). Forecasting of lead-time demand variance: Implications for safety stock calculations
- E Bayraktar, M Ludkovski (2010). Inventory management with partially observed nonstationary demand
- R Boute, J Gijsbrechts, W Van Jaarsveld, N Vanvuchelen (2021). Deep reinforcement learning for inventory control: A roadmap
- Y Cao, Z Shen (2019). Quantile forecasting and data-driven inventory management under nonstationary demand
- H Dehaybe, D Catanzaro, P Chevalier (2024). Deep reinforcement learning for inventory optimization with non-stationary uncertain demand
- G Dulac-Arnold, N Levine, D Mankowitz, J Li, C Paduraru, S Gowal, T Hester (2021). Challenges of real-world reinforcement learning: definitions, benchmarks and analysis
- K Geevers, L Van Hezewijk, M Mes (2023). Multi-echelon inventory optimization using deep reinforcement learning. Central
- J Gijsbrechts, R Boute, J Van Mieghem, D Zhang (2022). Can deep reinforcement learning improve inventory management? Performance on lost sales, dual-sourcing, and multi-echelon problems
- S Graves (1999). A single-item inventory model for a nonstationary demand process
- J Van Der Haar, W Van Jaarsveld, R Basten, R Boute (2024). Industrializing deep reinforcement learning for operational spare parts inventory management
- J Hu, C Zhang, C Zhu (2016). Inventory systems with correlated demands
- R Hyndman, A Koehler, R Snyder, S Grose (2002). A state space framework for automatic forecasting using exponential smoothing methods
- I Kaynov, M Van Knippenberg, V Menkovski, A Van Breemen, W Van Jaarsveld (2024). Deep reinforcement learning for one-warehouse multi-retailer inventory management
- X Ma, R Rossi, T Archibald (2022). Approximations for non-stationary stochastic lot-sizing under (s,Q)-type policy
- D Prak, R Teunter (2019). A general method for addressing forecasting uncertainty in inventory models
- A Shakya, G Pillai, S Chakrabarty (2023). Reinforcement learning algorithms: A brief survey
- J.-S Song, P Zipkin (1993). Inventory control in a fluctuating demand environment
- F Stranieri, E Fadda, F Stella (2024). Combining deep reinforcement learning and multi-stage stochastic programming to address the supply chain inventory management problem
- L Strijbosch, A Syntetos, J Boylan, E Janssen (2011). On the interaction between forecasting and stock control: The case of non-stationary demand
- T Temizöz, C Imdahl, R Dijkman, D Lamghari-Idrissi, W Van Jaarsveld (2023). Deep controlled learning for inventory control
- T Temizöz, C Imdahl, R Dijkman, D Lamghari-Idrissi, W Van Jaarsveld (2024). Zero-shot generalization in inventory management: Train, then estimate and decide
- H Tunc, O Kilic, S Tarim, B Eksioglu (2011). The cost of using stationary inventory policies when demand is non-stationary
- T Van Dijck, T Fleuren, T Temizoz, Y Merzifonluoglu, M Hendriks, W Van Jaarsveld (2024). Inventory planning in capacitated high-tech assembly systems under non-stationary demand
- L Van Hezewijk, N Dellaert, W Van Jaarsveld (2023). A new discrete non-stationary demand process with applications in inventory control
- L Van Hezewijk, N Dellaert, T Van Woensel, N Gademann (2023). Using the proximal policy optimisation algorithm for solving the stochastic capacitated lot sizing problem
- N Vanvuchelen, B De Moor, R Boute (2025). The use of continuous action representations to scale deep reinforcement learning for inventory control
- N Vanvuchelen, J Gijsbrechts, R Boute (2020). Use of proximal policy optimization for the joint replenishment problem
- P Verleijsdonk, W Van Jaarsveld, S Kapodistria (2024). Scalable policies for the dynamic traveling multi-maintainer problem with alerts


---

## Appendix — Tables, Algorithm, Figures (s2orc 추출 보강)

> 원본 markdown 추출이 누락한 표·알고리즘·그림 캡션을 s2orc JSON에서 복원. paper2code 구현에 필수.

### Table 1 — MDP notation

| 기호 | 정의 |
| --- | --- |
| $I_{i,t}$ | 제품 $i$, 기간 $t$ 시작 시점 재고 위치 |
| $[I_{i,t}]^+ = \max(I_{i,t},0)$ | 제품 $i$, 기간 $t$ 보유 재고(on-hand) |
| $[I_{i,t}]^- = -\min(I_{i,t},0)$ | 제품 $i$, 기간 $t$ 백오더 |
| $q_{i,t} \in \mathbb{Z}$ | 제품 $i$, 기간 $t$ 생산량 |
| $C \in \mathbb{Z}_{>0}$ | 기간당 최대 가용 capacity (시간 단위) |
| $\tau \in \mathbb{Z}_{\ge 0}$ | 기간 $t$ 내 사용된 capacity |
| $z_{i,t} \in \{0,1\}$ | 제품 $i$, 기간 $t$ 셋업 수행 지시자 |
| $\omega_i \in \{0,1\}$ | 머신이 제품 $i$로 셋업되어 있는지 지시 |
| $\theta_i \in \mathbb{Z}_{\ge 0}$ | 제품 $i$ 셋업 소요 시간 단위 |
| $\mu_{i,t} \in \mathbb{R}_{\ge 0}$ | 제품 $i$, 기간 $t$ 예측 수요(평균) |
| $\sigma_{i,t} \in \mathbb{R}_{\ge 0}$ | 제품 $i$, 기간 $t$ 예측 수요 표준편차 |
| $d_{i,t} \in \mathbb{Z}_{\ge 0}$ | 제품 $i$, 기간 $t$ 관측 수요 |
| $h_i$ | 제품 $i$ 기간당 보유비 |
| $b_i$ | 제품 $i$ 기간당 백오더비 |
| $k_i$ | 제품 $i$ 셋업비 |

### Table 2 — DCL 알고리즘 하이퍼파라미터

| $K$ | Nodes & layers | $N$ (샘플) | $M$ (액션당 롤아웃) | $H$ (롤아웃 horizon) | Generations |
| --- | --- | --- | --- | --- | --- |
| 5 | [128, 128] | 150,000 | 1000 | 30 | 5 |
| 10, 15 | [256, 256] | 300,000 | 1000 | 30 | 5 |

### Algorithm 1 — Rollout policy (AMBS 기반, 의사코드 재구성)

파라미터: $B_{min}$, $H_{max}$, $Z_{max}$ (그리드 서치로 결정). $\omega$ = 현재 셋업된 제품.

```
모든 제품 i∈K 의 reorder level까지 스케일드 갭 계산:
  gs_{i,t} = max( μ_{i,t} + B_min·σ_{i,t} − I_{i,t}, 0 )

while capacity 남음 (τ < C) and 생산 중단 안됨 (A1_τ ≠ p0):
    # 1단계 액션 A1_τ: 생산할 제품 결정
    if reorder level에 갭 있는 제품 존재 ( max_i gs_{i,t} > 0 )
       and 신규 셋업 허용 ( Σ_i z_{i,t} < Z_max ):
        가장 큰 갭 제품 생산: A1_τ = p_i,  i = argmax_i gs_{i,t}
        해당 갭 리셋: gs_{i,t} = 0
    else if 현재 셋업 제품 ω 의 재고가 한계 미만
            ( I_{ω,t} < μ_{ω,t} + B_min·σ_{ω,t} ):
        현재 셋업 제품 계속 생산: A1_τ = p_ω
    else:
        생산 중단: A1_τ = p0

    # 2단계 액션 A2_τ: 생산량 결정 (order-up-to, EOQ 항 포함, capacity 한계)
    A2_τ = min( ⌈ μ_{ω,t} + sqrt( 2·μ_{ω,t}·k_ω / h_ω ) + H_max·B_min·σ_{ω,t} ⌉ − I_{ω,t},  C − τ )
```

> 주의: A2 의 order-up-to 상한 항(`sqrt(2·μ·k/h)` EOQ 성분, `H_max·B_min·σ` 안전재고 성분)은 s2orc 테이블 셀 분할로 일부 재배열됨. 구현 전 원문 Algorithm 1 이미지 대조 권장.

### Table 3 — 정상(stationary) 설정, van Hezewijk(2023b) 실험 결과 (기간당 평균비용 + Δ)

| $K$ | $D_t$ | $f_c$ | Rollout | AMBS | PPO | DCL | $\Delta_{Rollout}$ | $\Delta_{PPO}$ | $\Delta_{DCL}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | U{3,5} | 1.1 | 41.02 | 36.63 | 35.14 | 33.22 | 12.0% | −4.1% | −9.3% |
| 5 | U{3,5} | 1.5 | 39.39 | 36.07 | 33.96 | 32.15 | 9.2% | −5.8% | −10.9% |
| 5 | U{0,8} | 1.1 | 41.83 | 39.06 | 37.73 | 35.60 | 7.1% | −3.4% | −8.9% |
| 5 | U{0,8} | 1.5 | 39.73 | 36.84 | 36.23 | 33.89 | 7.8% | −1.7% | −8.0% |
| 10 | U{3,5} | 1.1 | 38.45 | 37.63 | 39.31 | 36.63 | 2.2% | 4.4% | −2.7% |
| 10 | U{3,5} | 1.5 | 38.51 | 37.31 | 39.86 | 35.33 | 3.2% | 6.9% | −5.3% |
| 10 | U{0,8} | 1.1 | 40.15 | 39.44 | 42.42 | 36.57 | 1.8% | 7.6% | −7.3% |
| 10 | U{0,8} | 1.5 | 39.87 | 38.23 | 41.25 | 35.91 | 4.3% | 7.9% | −6.1% |
| 15 | U{3,5} | 1.1 | 39.95 | 38.33 | − | 37.09 | 4.2% | − | −3.2% |
| 15 | U{3,5} | 1.5 | 39.14 | 37.66 | − | 36.44 | 3.9% | − | −3.2% |
| 15 | U{0,8} | 1.1 | 40.98 | 40.53 | − | 37.70 | 1.1% | − | −7.0% |
| 15 | U{0,8} | 1.5 | 39.62 | 38.65 | − | 36.83 | 2.5% | − | −4.7% |

> Δ = AMBS 대비 차이(음수=DCL/PPO 우위). PPO는 15제품에서 intractable(−).

### Table 4 — 비정상(non-stationary) 실험 설정

| 파라미터 | 값 |
| --- | --- |
| Backorder cost $b_i$ | 9 |
| Holding cost $h_i$ | 1 |
| Setup cost $k_i$ | 100 |
| Setup time $\theta_i$ | 1 |
| Initial mean demand $\mu_{i,0}$ | 2 |
| Initial COV $COV_{i,0}$ | 0.5, 1.0 |
| Non-stationarity $\alpha$ | 0, 0.025, 0.05 |
| Number of products $K$ | 5, 10, 15 |
| Capacity factor $f_c$ | 1.5, 2.5 |
| Total experiments | 36 |

> **주의(파라미터 충돌)**: §5.1 정상 설정은 van Hezewijk 복제용으로 $k_i=200$, $\theta_i=0$ 사용. 비정상 실험(Table 4)은 $k_i=100$, $\theta_i=1$. 실험군별 상이 — 코드에서 분리할 것.

### Table 5 — 비정상 실험 결과 (DCL의 AMBS 대비 Δ)

| $f_c$ | $COV_{i,0}$ | $\alpha$ | $\Delta_{K=5}$ | $\Delta_{K=10}$ | $\Delta_{K=15}$ |
| --- | --- | --- | --- | --- | --- |
| 1.5 | 0.5 | 0 | −9.4% | −6.1% | −6.2% |
| 1.5 | 0.5 | 0.025 | −8.6% | −7.2% | −6.9% |
| 1.5 | 0.5 | 0.05 | −8.0% | −7.1% | −6.4% |
| 1.5 | 1.0 | 0 | −7.4% | −6.1% | −3.9% |
| 1.5 | 1.0 | 0.025 | −6.8% | −6.1% | −3.2% |
| 1.5 | 1.0 | 0.05 | −8.3% | −7.9% | −2.7% |
| 2.5 | 0.5 | 0 | −13.6% | −6.8% | −0.3% |
| 2.5 | 0.5 | 0.025 | −13.6% | −7.1% | −2.8% |
| 2.5 | 0.5 | 0.05 | −13.0% | −6.8% | −3.5% |
| 2.5 | 1.0 | 0 | −11.3% | −6.1% | −3.7% |
| 2.5 | 1.0 | 0.025 | −11.2% | −6.1% | −3.6% |
| 2.5 | 1.0 | 0.05 | −10.9% | −6.2% | −3.0% |

### Table 6 — DCL 학습 계산 시간 (4 병렬 노드, AMD Genoa 9654)

| $K$ | Samples | Training (min/gen) | Sample collection (min/gen) | Total (hrs) |
| --- | --- | --- | --- | --- |
| 5 | 150,000 | 2 | 6 | 0.67 |
| 10 | 300,000 | 7 | 40 | 3.92 |
| 15 | 300,000 | 10 | 90 | 8.33 |

### Figure 캡션

- **Fig. 1** — van Hezewijk et al. (2023b) MDP의 전이 동역학(full-period 결정).
- **Fig. 2** — 본 논문 신규 MDP 전이 동역학. 점선 = 기간 내 sub-decision 전이.
- **Fig. 3** — trajectory(T=100)에서 일일 수요가 일일 capacity 초과한 평균 기간 수. $t=1$, $C = 1.5\cdot\sum_{i}\mu_{i,0}$.
- **Fig. 4** — horizon $T$ 누적 수요가 누적 capacity 초과한 trajectory 수(10,000 중).
- **Fig. 5** — AMBS vs DCL 정책의 재고/셋업/백오더 분해 (본문 §5.3).