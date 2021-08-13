from .base_options import BaseOptions, str2bool


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.isTrain = True
        return parser
