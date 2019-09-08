from pandas import DataFrame

def data_dict(data_frame: DataFrame) -> dict:
    return {
        'sentiment': {
            'type': data_frame.sentiment.dtype,
            'description': 'sentiment class - 0:negative, 1:positive'
        },
        'text': {
            'type': data_frame.text.dtype,
            'description': 'tweet text'
        },
        'pre_clean_len': {
            'type': data_frame.pre_clean_len.dtype,
            'description': 'length of the tweet before cleaning'
        },
        'dataset_shape': data_frame.shape
    }