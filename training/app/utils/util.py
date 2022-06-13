from datetime import datetime


def name_model(ds) -> str:
    # Since there is no versioning in S3, let's prefix model with day + time
    now = datetime.now()
    postfix = now.strftime("%m.%d:%H.%M")
    name = f'{ds.name}_la_{postfix}'
    print(f'Named model {name}')
    return name
