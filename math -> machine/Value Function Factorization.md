- **What it is:** A popular CTDE technique that the paper (and your group 17171717) discusses. It's the foundation that A-QMIX and COPA build upon.
    
- **The Idea:** It learns the team's total Q-value (the expected future reward, $Q^{tot}$) by "factorizing" it into _individual_ Q-values ($Q^a$) for each agent18.
    
- **The "Monotonicity Constraint":** This is the key trick, which your group discussed191919191919191919. QMIX enforces a constraint that $\frac{\partial Q_{tot}}{\partial Q^{a}}\ge0$20. In simple terms, this means that an agent improving its _own_ Q-value will _never_ decrease the _team's_ Q-value.
    
- **The Benefit:** This guarantees that during decentralized execution, if each agent greedily picks the action that maximizes its _own_ $Q^a$, the resulting _joint_ action will also maximize the _team's_ $Q^{tot}$21. This is exactly what your group concluded: "the optimal action from each individual agent is also the bisection best action for the team"