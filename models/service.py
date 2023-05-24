class Service:
    def __init__(self):
        self.name = 'model'
        pass

    def get_model_name(self) -> str:
        return self.name

    def load_model(self, model: bytes):
        pass

    def _transform_in(self, input):
        return input

    def _transform_out(self, result):
        return result

    def _process_batch(self, data):
        return data

    def process(self, inputs: [(bytes, bytes)]):
        if len(inputs) <= 20:
            transformed = [self._transform_in(input) for input in inputs]
            filtered = [input for input in transformed if input is not None]
            results = [self._transform_out(result) for result in self._process_batch(filtered)]

            i = 0
            for j in range(len(transformed)):
                if transformed[j] is not None:
                    transformed[j] = results[i]
                    i += 1

            return [bytes() if result is None else result for result in transformed]

        results = []
        for i in range(0, len(inputs), 20):
            results += self.process(inputs[i:i + 20])
        return results
