class ArmException(Exception):
    pass


class TimeCanNotBeNegative(ArmException):
    pass


class TimeStepCanNotExceedTmax(ArmException):
    pass


class ObservationNumberDoesNotMatch(ArmException):
    pass


# Bernoulli
class BernoulliException(ArmException):
    pass


class BernoulliProbabilityBeyondBounds(BernoulliException):
    pass


class BernoulliBounds(BernoulliException):
    pass


# Uniform
class UniformException(ArmException):
    pass


class UnifromBounds(UniformException):
    pass
