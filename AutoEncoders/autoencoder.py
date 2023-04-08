from abc import ABCMeta, abstractmethod


class AutoEncoder(metaclass=ABCMeta):
    """ Base face detector class. """

    @abstractmethod
    def compile(self, optimizer, loss):
        pass
