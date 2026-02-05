formulas = []

items = ['pickaxe', 'lava', 'door', 'gem', 'empty' ]

#PATTERNS INSPIRED FROM LTL2action
# Formulas have pattern LTL, number of different items, textual description
formulas.append(("(F c0) & (F c1)", 2, "task1: visit({0}, {1})".format(*items)))
formulas.append(("(F c0) & (F c1) & (F c2)", 3, "task2: visit({0}, {1}, {2})".format(*items)))
formulas.append(("F(c0 & F(c1))", 2, "task3: seq_visit({0}, {1})".format(*items)))
formulas.append(("F(c0 & F(c1)) & F(c2 & F(c3))", 4, "task4: seq_visit({0}, {1}) + seq_visit({2}, {3})".format(*items)))
formulas.append(("F(c0 & F(c1)) & (F c2)", 3, "task5: seq_visit({0}, {1}) + visit({2})".format(*items)))
formulas.append(("F(c0 & F(c1)) & (F c2) & (F c3)", 4, "task6: seq_visit({0}, {1}) + visit({2}, {3})".format(*items)))
formulas.append(("(F c0) & (F c1) & (G (! c2))", 3, "task7: visit({0}, {1}) + glob_av({2})".format(*items)))
formulas.append(("(F c0) & (F c1) & (G (! c2)) & (G(! c3))", 4, "task8: visit({0}, {1}) + glob_av({2}) + glob_av({3})".format(*items)))
formulas.append(("F(c0 & F(c1)) & G (! c2)", 3, "task9: seq_visit({0}, {1}) + glob_av({2})".format(*items)))
formulas.append(("F(c0 & F(c1)) & G (! c2) & G(! c3)", 4, "task10: seq_visit({0}, {1}) + glob_av({2}) + glob_av({3})".format(*items)))

# Utterances corresponding to the formulas.
utterances = []
utterances.append("Visit the pickaxe and visit the lava.")
utterances.append("Visit the pickaxe, the lava, and the door.")
utterances.append("First visit the pickaxe, then visit the lava.")
utterances.append("First visit the pickaxe, then visit the lava. After that, first visit the door, then visit the gem.")
utterances.append("First visit the pickaxe, then visit the lava. Also, visit the door.")
utterances.append("First visit the pickaxe, then visit the lava. Also, visit the door and the gem.")
utterances.append("Visit the pickaxe and the lava, always avoid the door.")
utterances.append("Visit the pickaxe and the lava, always avoid the door and the gem.")
utterances.append("First visit the pickaxe, then visit the lava. Always avoid the door.")
utterances.append("First visit the pickaxe, then visit the lava. Always avoid the door and the gem.")

# LTLs obtained by prompting the LLM.
ltls = []
ltls.append("F c0 & F c1")
ltls.append("F(c0 & F(c1 & F(c2)))")
ltls.append("F(c0 & F(c1))")
ltls.append("F(c0 & F(c1 & F(c2 & F(c3))))")
ltls.append("F(c0 & F(c1)) & F(c2)")
ltls.append("F(c0 & F(c1)) & F(c2) & F(c3)")
ltls.append("F(c0 & F c1) & G(!c2)")
ltls.append("F(c0 & F c1) & G(!c2 & !c3)")
ltls.append("F(c0 & F(c1)) & G(!c2)")
ltls.append("F(c0 & F(c1)) & G(!c2 & !c3)")

# Sample formula.
formula = ("(F c0) & (F c1)", 2, "task0: visit(pickaxe, lava)")
formula1 = ("F(c0 & F(c1)) & (F c2) & (F c3)", 4, "task6: seq_visit(pickaxe, lava) + visit(door, gem)")
formula2 = ("(F c0) & (F c1) & (F c2)", 3, "task2: visit(pickaxe, lava, door)")
formula10 = ("F(c0 & F(c1)) & G (! c2) & G(! c3)", 4, "task10: seq_visit(pickaxe, lava) + glob_av(door) + glob_av(gem)")