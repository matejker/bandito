class PolicyException(Exception):
    pass


class TimeCanNotBeNegative(PolicyException):
    pass


class UniformPolicy(PolicyException):
    pass
