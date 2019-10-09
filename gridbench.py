#!/usr/bin/env python3


class OptimizationDimension:
    pass


class LinearDimension(OptimizationDimension):
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi


class FixedValues(OptimizationDimension):
    def __init__(self, ls):
        self.ls = ls


class StringArgumentFactory:
    def __init__(self, argument_template):
        self.template = argument_template

    def get_arg(self, *value):
        return self.template.format(*value)
