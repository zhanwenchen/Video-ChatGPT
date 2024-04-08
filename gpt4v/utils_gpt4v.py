from datetime import datetime as datetime_datetime


def generate_indexName():
    time_str = datetime_datetime.now().strftime('%Y%m%d%H%M%S')
    return f'vtomindex{time_str}'
