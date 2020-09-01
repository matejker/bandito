class ArmException(Exception):
    pass


class TimeCanNotBeNegative(ArmException):
    pass


class TimeStepCanNotExceedTmax(ArmException):
    pass


class ObservationNumberDoesNotMatch(ArmException):
    pass
