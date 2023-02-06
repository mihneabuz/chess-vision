class Service:
    def __init__(self):
        self.name = 'model'
        pass

    def get_model_name(self):
        return self.name

    def load_model(self, model):
        pass

    def _transform_in(self, input):
        return input

    def _transform_out(self, result):
        return result

    def _process_batch(self, data):
        return data

    def process(self, inputs):
        results = self._process_batch([self._transform_in(input) for input in inputs])
        return [self._transform_out(result) for result in results]
