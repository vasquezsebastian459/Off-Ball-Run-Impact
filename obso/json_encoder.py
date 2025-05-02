

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
        
    # Assuming 'run' is a dictionary containing your data
    run = {
        "value": np.float32(1.0)  # Example of a float32 value
    }

    # Use the custom encoder
    json_data = json.dumps(run, cls=NumpyEncoder)
    print(json_data)